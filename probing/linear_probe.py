#!/usr/bin/env python3
"""
Linear probe script for evaluating saved models from Synthetic_Real_World_Models.
This script loads a PEFT model, extracts the base model, and trains a linear classifier
on top to detect template bias.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pandas as pd
from google.cloud import storage
import io
import os
import argparse
import numpy as np
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ClinicalNotesDataset(Dataset):
    """Custom dataset class for handling clinical notes."""
    
    def __init__(self, texts: list[str], labels, tokenizer, max_length=512):
        # Tokenize input texts with padding and truncation
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __getitem__(self, idx):
        # Return encoding features and label for a specific index
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
        
    def __len__(self):
        # Return the total number of samples
        return len(self.labels)


class LinearProbeClassifier(nn.Module):
    """Linear classifier on top of frozen base model."""
    
    def __init__(self, base_model, num_labels):
        """
        Initialize linear probe classifier with pre-trained base model.
        Args:
            base_model: Pre-trained transformer model (frozen)
            num_labels: Number of output classes
        """
        super().__init__()
        self.base_model = base_model
        # Get padding token ID (or use EOS token if pad token is not defined)
        self.pad_token_id = base_model.config.pad_token_id or base_model.config.eos_token_id
        
        # Freeze the base model's parameters for transfer learning
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Classification head - linear layer that takes hidden states to output logits
        hidden_size = base_model.config.hidden_size
        # Get the dtype from the base model to ensure compatibility
        # Check the dtype of the first parameter
        base_dtype = next(base_model.parameters()).dtype
        self.classifier = nn.Linear(hidden_size, num_labels, dtype=base_dtype)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        Args:
            input_ids: Token IDs of input text
            attention_mask: Mask indicating which tokens are padding
        Returns:
            logits: Raw prediction scores
        """
        with torch.no_grad():
            hidden_states = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state

        # Find the last non-padding token for each sequence using attention_mask
        # attention_mask.sum(dim=1) gives sequence length, subtract 1 to get last index
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_non_pad_index = seq_lengths.clamp(min=0)  # Ensure non-negative indices

        # Select the hidden state of the last non-pad token
        pooled_hidden = hidden_states[torch.arange(input_ids.size(0)), last_non_pad_index]
        
        # Ensure pooled_hidden is in the same dtype as the classifier
        # (This handles the case where base model is bfloat16 but classifier might be float32)
        pooled_hidden = pooled_hidden.to(self.classifier.weight.dtype)
        
        # Compute logits
        logits = self.classifier(pooled_hidden)
        return logits


def load_data_from_gcs(bucket_name, blob_name):
    """Load data from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    data = blob.download_as_string()
    df = pd.read_csv(io.BytesIO(data))
    
    return df


def combine_template_csvs(template_a_path, template_b_path, text_column='full_note', bucket_name=None, max_samples_per_template=None, random_seed=42):
    """
    Combine two template CSV files (template A and template B) into a single dataframe
    suitable for linear probe evaluation.
    
    Args:
        template_a_path: Path to template A CSV (local file or GCS blob name)
        template_b_path: Path to template B CSV (local file or GCS blob name)
        text_column: Name of the column containing the text (default: 'full_note')
        bucket_name: GCS bucket name (if paths are GCS blobs, otherwise None for local files)
        max_samples_per_template: Maximum number of samples to use from each template (None = use all)
        random_seed: Random seed for sampling (default: 42)
    
    Returns:
        Combined DataFrame with 'template' column ('A' or 'B') and text column
    """
    # Load template A - check local first, then GCS if bucket_name provided
    if os.path.exists(template_a_path):
        df_a = pd.read_csv(template_a_path)
    elif bucket_name:
        # Try GCS if file doesn't exist locally and bucket_name is provided
        df_a = load_data_from_gcs(bucket_name, template_a_path)
    else:
        raise FileNotFoundError(f"Template A file not found locally at {template_a_path} and no bucket_name provided for GCS")
    
    # Load template B - check local first, then GCS if bucket_name provided
    if os.path.exists(template_b_path):
        df_b = pd.read_csv(template_b_path)
    elif bucket_name:
        # Try GCS if file doesn't exist locally and bucket_name is provided
        df_b = load_data_from_gcs(bucket_name, template_b_path)
    else:
        raise FileNotFoundError(f"Template B file not found locally at {template_b_path} and no bucket_name provided for GCS")
    
    # Sample if max_samples_per_template is specified
    if max_samples_per_template is not None:
        original_len_a = len(df_a)
        if original_len_a > max_samples_per_template:
            df_a = df_a.sample(n=max_samples_per_template, random_state=random_seed, replace=False)
            print(f"Sampled {max_samples_per_template} from Template A (had {original_len_a} total)")
        else:
            print(f"Using all {original_len_a} Template A samples (less than {max_samples_per_template} requested)")
        
        original_len_b = len(df_b)
        if original_len_b > max_samples_per_template:
            df_b = df_b.sample(n=max_samples_per_template, random_state=random_seed, replace=False)
            print(f"Sampled {max_samples_per_template} from Template B (had {original_len_b} total)")
        else:
            print(f"Using all {original_len_b} Template B samples (less than {max_samples_per_template} requested)")
    
    # Add template identifier
    df_a['template'] = 'A'
    df_b['template'] = 'B'
    
    # Combine
    combined_df = pd.concat([df_a, df_b], ignore_index=True)
    
    # Shuffle the combined dataframe
    combined_df = combined_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Ensure the text column exists (rename if needed for consistency)
    if text_column not in combined_df.columns:
        # Try to find a common text column name
        possible_cols = ['full_note', 'text', 'note', 'reformatted_text']
        for col in possible_cols:
            if col in combined_df.columns:
                combined_df[text_column] = combined_df[col]
                break
        else:
            raise ValueError(f"Could not find text column '{text_column}' or any common text column in the data")
    elif text_column != 'reformatted_text':
        # Create reformatted_text column for consistency with default
        combined_df['reformatted_text'] = combined_df[text_column]
    
    print(f"Combined {len(df_a)} Template A samples + {len(df_b)} Template B samples = {len(combined_df)} total samples")
    
    return combined_df


def load_data_from_gcs_flatten(bucket_name, blob_name, use_raw_text=False):
    """Load data from Google Cloud Storage and flatten template structure."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    data = blob.download_as_string()
    original_df = pd.read_csv(io.BytesIO(data))
    
    # Create two rows for each original row (one for each template)
    template_rows = []
    
    for _, row in original_df.iterrows():
        # Determine what text to use based on argument
        if use_raw_text:
            template_a_text = row['text']
            template_b_text = row['text']
        else:
            template_a_text = row['formatted_text_templateA']
            template_b_text = row['formatted_text_templateB']
        
        # Create row for template A
        template_rows.append({
            'note_id': row['note_id'],
            'note': row['text'],  # Original note text
            'reformatted_text': template_a_text,
            'template': 'A',
            'label': row['label']
        })
        
        # Create row for template B
        template_rows.append({
            'note_id': row['note_id'],
            'note': row['text'],  # Original note text
            'reformatted_text': template_b_text,
            'template': 'B',
            'label': row['label']
        })
    
    # Create new dataframe with template structure
    reformatted_df = pd.DataFrame(template_rows)
    
    return reformatted_df


def load_peft_model(model_path):
    """Load a PEFT model and return the base model (without classification head)."""
    print(f"Loading PEFT model from: {model_path}")
    
    # Load PEFT config to get base model name
    config = PeftConfig.from_pretrained(model_path)
    print(f"Base model: {config.base_model_name_or_path}")
    
    # Load base model for sequence classification
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=2,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    # Load PEFT adapter
    peft_model = PeftModel.from_pretrained(base_model, model_path)
    
    # Extract the base transformer model (without classification head)
    # For Llama models with PEFT, structure is: PeftModel -> base_model -> model -> model (LlamaModel)
    # Try different paths to find the actual transformer model
    if hasattr(peft_model, 'base_model') and hasattr(peft_model.base_model, 'model'):
        if hasattr(peft_model.base_model.model, 'model'):
            # LlamaForSequenceClassification -> model -> model (LlamaModel)
            base_transformer = peft_model.base_model.model.model
        else:
            # Fallback: use base_model.model
            base_transformer = peft_model.base_model.model
    elif hasattr(peft_model, 'model'):
        if hasattr(peft_model.model, 'model'):
            base_transformer = peft_model.model.model
        else:
            base_transformer = peft_model.model
    else:
        raise ValueError("Could not extract base transformer model from PEFT model")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return base_transformer, tokenizer


def train_model(model, train_dataloader, val_dataloader, num_epochs=3, learning_rate=2e-5):
    """Train the linear probe model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_val_auc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        all_labels = []
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for AUC calculation
                probs = torch.nn.functional.softmax(outputs, dim=1)
                # Convert to float32 before numpy conversion (numpy doesn't support bfloat16)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().float().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        # Calculate AUC - handle multiclass case
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        
        # For binary classification
        if all_probs.shape[1] == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        # For multi-class classification
        else:
            # One-vs-Rest AUC for each class
            auc_scores = []
            for i in range(all_probs.shape[1]):
                # Create binary labels for this class
                binary_labels = (all_labels == i).astype(int)
                # AUC for this class vs rest
                try:
                    class_auc = roc_auc_score(binary_labels, all_probs[:, i])
                    auc_scores.append(class_auc)
                except ValueError:
                    # This can happen if a class is not present in this batch
                    continue
            
            # Average AUC across all classes
            auc = np.mean(auc_scores) if auc_scores else 0.0
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Training Loss: {train_loss/len(train_dataloader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_dataloader):.4f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')
        print(f'Validation AUC: {auc:.4f}')
        
        if auc > best_val_auc:
            best_val_auc = auc
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
    
    return best_val_auc, all_labels, all_preds, all_probs


def main():
    parser = argparse.ArgumentParser(description='Run linear probe on saved models')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to saved model (e.g., Synthetic_Real_World_Experiment/Synethtic_Real_World_Models/p_0_final_model)')
    parser.add_argument('--data-source', type=str, choices=['gcs', 'gcs_flatten', 'local', 'combine_templates', 'presplit'],
                        default='gcs_flatten',
                        help='Data source: gcs (load from GCS), gcs_flatten (load from GCS and create template rows), local (load from local file), combine_templates (combine two template CSV files), or presplit (use pre-split train/val/test parquet files from process_data.py)')
    parser.add_argument('--bucket-name', type=str, default=None,
                        help='GCS bucket name (only used if data-source is gcs, gcs_flatten, or if local files not found for combine_templates)')
    parser.add_argument('--blob-name', type=str,
                        default=None,
                        help='For GCS: path to file in bucket (e.g., "Reformatted_Notes/file.csv"). For local: path to CSV file. Required columns for gcs_flatten: formatted_text_templateA, formatted_text_templateB, note_id, text, label')
    parser.add_argument('--local-data-path', type=str,
                        help='Local CSV file path (alternative to --blob-name when data-source is local)')
    parser.add_argument('--train-data', type=str,
                        help='Path to train.parquet file (for presplit data-source)')
    parser.add_argument('--val-data', type=str,
                        help='Path to val.parquet file (for presplit data-source)')
    parser.add_argument('--template-a-path', type=str,
                        help='Path to template A CSV (for combine_templates data source)')
    parser.add_argument('--template-b-path', type=str,
                        help='Path to template B CSV (for combine_templates data source)')
    parser.add_argument('--max-samples-per-template', type=int, default=None,
                        help='Maximum number of samples to use from each template (for combine_templates, default: None = use all)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    parser.add_argument('--text-column', type=str, default='reformatted_text',
                        help='Column name for text data (default: reformatted_text, but use "full_note" if using data from finetune_with_synthetic_bias.py)')
    parser.add_argument('--template-column', type=str, default='template',
                        help='Column name for template labels')
    parser.add_argument('--num-epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate for linear probe')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max-length', type=int, default=9000,
                        help='Maximum sequence length')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Validation set size (fraction)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (optional)')
    
    args = parser.parse_args()
    
    # Load model
    print("="*70)
    print("Loading Model")
    print("="*70)
    base_model, tokenizer = load_peft_model(args.model_path)
    print("Model loaded successfully!")
    
    # Load data
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)

    if args.data_source == 'presplit':
        # Load pre-split train/val/test parquet files from process_data.py
        if not args.train_data or not args.val_data:
            raise ValueError("--train-data and --val-data are required when using presplit data source")

        print(f"Loading pre-split data:")
        print(f"  Train: {args.train_data}")
        print(f"  Val: {args.val_data}")

        train_df = pd.read_parquet(args.train_data)
        val_df = pd.read_parquet(args.val_data)

        # process_data.py outputs 'text' and 'template' columns
        train_df = train_df.dropna()
        val_df = val_df.dropna()

        # Extract texts and labels
        train_texts = train_df['text'].tolist()
        val_texts = val_df['text'].tolist()

        # Map template labels to binary (A=1, B=0)
        label_map = {"A": 1, "B": 0}
        train_labels = [label_map[label] for label in train_df['template'].tolist()]
        val_labels = [label_map[label] for label in val_df['template'].tolist()]

        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        print(f"Train template distribution: {train_df['template'].value_counts().to_dict()}")
        print(f"Val template distribution: {val_df['template'].value_counts().to_dict()}")

        # Create datasets directly (no splitting needed)
        train_dataset = ClinicalNotesDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
        val_dataset = ClinicalNotesDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)

    else:
        # Original data loading logic for other sources
        if args.data_source == 'local':
            if args.local_data_path:
                df = pd.read_csv(args.local_data_path)
            else:
                df = pd.read_csv(args.blob_name)
        elif args.data_source == 'gcs_flatten':
            df = load_data_from_gcs_flatten(args.bucket_name, args.blob_name)
        elif args.data_source == 'combine_templates':
            if not args.template_a_path or not args.template_b_path:
                raise ValueError("--template-a-path and --template-b-path are required when using combine_templates data source")

            # Check if files exist locally first
            template_a_exists = os.path.exists(args.template_a_path)
            template_b_exists = os.path.exists(args.template_b_path)

            if not template_a_exists and not template_b_exists:
                # Neither file exists locally - check if we should try GCS
                if args.bucket_name:
                    print(f"Warning: Template files not found locally. Trying GCS...")
                    print(f"  Template A: {args.template_a_path}")
                    print(f"  Template B: {args.template_b_path}")
                    use_gcs = True
                else:
                    raise FileNotFoundError(
                        f"Template files not found locally and no bucket_name provided.\n"
                        f"  Template A: {args.template_a_path}\n"
                        f"  Template B: {args.template_b_path}\n"
                        f"Please provide local file paths or set --bucket-name for GCS."
                    )
            elif not template_a_exists or not template_b_exists:
                # One file exists but not the other
                missing = []
                if not template_a_exists:
                    missing.append(f"Template A: {args.template_a_path}")
                if not template_b_exists:
                    missing.append(f"Template B: {args.template_b_path}")
                raise FileNotFoundError(
                    f"Some template files not found locally:\n  " + "\n  ".join(missing) +
                    f"\nPlease ensure all template files exist locally or provide correct paths."
                )
            else:
                # Both files exist locally - use local files
                print(f"Using local template files:")
                print(f"  Template A: {args.template_a_path}")
                print(f"  Template B: {args.template_b_path}")
                use_gcs = False

            df = combine_template_csvs(
                args.template_a_path,
                args.template_b_path,
                text_column=args.text_column,
                bucket_name=args.bucket_name if use_gcs else None,
                max_samples_per_template=args.max_samples_per_template,
                random_seed=args.random_seed
            )
        else:  # gcs
            df = load_data_from_gcs(args.bucket_name, args.blob_name)

        df = df.dropna()

        # Prepare texts and labels
        texts = df[args.text_column].tolist()
        label_map = {"A": 1, "B": 0}
        labels = df[args.template_column].tolist()
        labels = [label_map[label] for label in labels]

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=args.test_size, random_state=42
        )

        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")

        # Create datasets
        train_dataset = ClinicalNotesDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
        val_dataset = ClinicalNotesDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize and train model
    print("\n" + "="*70)
    print("Training Linear Probe")
    print("="*70)
    model = LinearProbeClassifier(base_model, num_labels=2)
    best_auc, val_labels, val_preds, val_probs = train_model(
        model, train_dataloader, val_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    # Save results if output directory is specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results = {
            'model_path': args.model_path,
            'best_val_auc': best_auc,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'train_samples': len(train_texts),
            'val_samples': len(val_texts),
        }
        
        results_file = os.path.join(args.output_dir, 'linear_probe_results.txt')
        with open(results_file, 'w') as f:
            f.write("Linear Probe Results\n")
            f.write("="*70 + "\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\nResults saved to {results_file}")
    
    print("\n" + "="*70)
    print(f"Best Validation AUC: {best_auc:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()

