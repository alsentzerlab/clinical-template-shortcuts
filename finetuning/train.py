#!/usr/bin/env python3
"""
Training Script for Flexible Fine-tuning
This script trains a model on preprocessed data.
It accepts any dataset via command line arguments.
"""

import torch
import pandas as pd
import os
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers.trainer_utils import get_last_checkpoint
import evaluate
from datetime import datetime

# Seed will be set after argument parsing

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def safe_auc(y_true, y_score):
    """roc_auc_score, but NaN instead of ValueError when only one class is present.

    At the sweep endpoints (p_train=0 or 1) every positive case is assigned to one
    template, leaving the other with zero positives. That template's AUC is genuinely
    undefined -- a property of the design, not a failure -- so report NaN and let the
    run finish rather than losing a completed model to an exception at metrics time.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def load_dataset(parquet_path, balanced_size=None):
    """Load dataset from parquet file.

    Args:
        parquet_path: Path to the parquet file
        balanced_size: If not None, sample this many samples with equal split between templates A and B
    """
    print(f"Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Filter out NaN values
    df = df.dropna(subset=['text', 'label'])

    # Convert labels to int if needed
    if df['label'].dtype != 'int64':
        df['label'] = df['label'].astype(int)

    # If balanced sampling is requested
    if balanced_size is not None and 'template' in df.columns:
        samples_per_template = balanced_size // 2
        print(f"Balancing dataset: {samples_per_template} samples per template")

        template_a = df[df['template'] == 'A'].sample(n=samples_per_template, random_state=42)
        template_b = df[df['template'] == 'B'].sample(n=samples_per_template, random_state=42)

        df = pd.concat([template_a, template_b], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

        print(f"Balanced to {len(df)} samples")
        print(f"Template A: {(df['template'] == 'A').sum()}, Template B: {(df['template'] == 'B').sum()}")

    print(f"Loaded {len(df)} samples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"Readmission rate: {df['label'].mean():.4f}")

    return df


def balance_template_readmission_rates(df, label_column='label'):
    """Equalize readmission rates between template A and B by downsampling.

    Keeps template A unchanged. Adjusts template B to match template A's
    readmission rate: downsamples B positives if r_B > r_A, or B negatives
    if r_B < r_A.
    """
    assert 'template' in df.columns, "DataFrame must have a 'template' column"

    df_a = df[df['template'] == 'A']
    df_b = df[df['template'] == 'B']

    r_a = df_a[label_column].mean()
    r_b = df_b[label_column].mean()

    print(f"Before balancing readmission rates: Template A={r_a:.4f}, Template B={r_b:.4f}")

    b_pos = df_b[df_b[label_column] == 1]
    b_neg = df_b[df_b[label_column] == 0]

    if r_b > r_a:
        # Downsample B positives: n_pos_new = r_a * n_neg / (1 - r_a)
        n_b_neg = len(b_neg)
        n_b_pos_new = int(round(r_a * n_b_neg / (1 - r_a)))
        n_b_pos_new = min(n_b_pos_new, len(b_pos))
        b_pos_sampled = b_pos.sample(n=n_b_pos_new, random_state=42)
        df_b_new = pd.concat([b_pos_sampled, b_neg], ignore_index=True)
    elif r_b < r_a:
        # Downsample B negatives: n_neg_new = n_pos * (1 - r_a) / r_a
        n_b_pos = len(b_pos)
        n_b_neg_new = int(round(n_b_pos * (1 - r_a) / r_a))
        n_b_neg_new = min(n_b_neg_new, len(b_neg))
        b_neg_sampled = b_neg.sample(n=n_b_neg_new, random_state=42)
        df_b_new = pd.concat([b_pos, b_neg_sampled], ignore_index=True)
    else:
        df_b_new = df_b

    df_b_new = df_b_new.sample(frac=1, random_state=42).reset_index(drop=True)
    result = pd.concat([df_a, df_b_new], ignore_index=True)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    r_b_new = df_b_new[label_column].mean()
    print(f"After balancing readmission rates: Template A={r_a:.4f}, Template B={r_b_new:.4f}")
    print(f"Template B size: {len(df_b)} -> {len(df_b_new)}")

    return result


def train_model(
    train_data_path,
    val_data_path,
    test_data_path,
    text_column='text',
    label_column='label',
    model_name='Qwen/Qwen3-0.6B',
    tokenizer_name=None,
    epochs=4,
    batch_size=1,
    per_device_batch_size=1,
    learning_rate=5e-5,
    output_dir='./models',
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    max_seq=None,
    gradient_checkpointing=False,
    wandb_project=None,
    wandb_entity=None,
    run_name=None,
    balanced_train_size=None,
    balanced_val_size=None,
    balanced_test_size=None,
    early_stopping_patience=2,
    balance_readmission_rates=False,
    seed=42
):
    """
    Train a classifier on preprocessed data.

    Args:
        train_data_path: Path to training parquet file
        val_data_path: Path to validation parquet file
        test_data_path: Path to test parquet file
        text_column: Name of column containing text
        label_column: Name of column containing labels
        model_name: HuggingFace model identifier
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        output_dir: Directory to save outputs
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        max_seq: Maximum sequence length for truncation (None = no truncation)
        wandb_project: W&B project name (optional)
        wandb_entity: W&B entity/username (optional)
        run_name: W&B run name (optional)
        balanced_train_size: If not None, balance train set to this size (half A, half B)
        balanced_val_size: If not None, balance val set to this size (half A, half B)
        balanced_test_size: If not None, balance test set to this size (half A, half B)
        early_stopping_patience: Number of epochs with no improvement before stopping
        balance_readmission_rates: If True, equalize template B readmission rate to match
            template A in the training split (downsample B positives if r_B > r_A,
            downsample B negatives if r_B < r_A)
        seed: Random seed for reproducibility
    """
    # Skip entirely if this run already completed (final model + results saved)
    final_model_dir = os.path.join(output_dir, "final_model")
    results_path = os.path.join(output_dir, 'results.txt')
    if os.path.isdir(final_model_dir) and os.path.isfile(results_path):
        print(f"Run already completed: found {final_model_dir} and {results_path}. Skipping training.")
        return None, None, None, None

    # Fail fast on a dead GPU instead of silently training on CPU (which just hangs).
    # Common on this cluster when a node's driver breaks (NVML mismatch).
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU visible (torch.cuda.is_available() is False). This node's GPU driver "
            "is likely broken (check `nvidia-smi`). Exit and srun a fresh node — refusing "
            "to train on CPU."
        )

    set_seed(seed)

    # Initialize W&B if requested
    if wandb_project:
        import wandb
        if run_name is None:
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "max_seq": max_seq,
            }
        )
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        if wandb_project:
            wandb.config.update({"cuda_device": torch.cuda.get_device_name(0)})
    
    # Load datasets
    train_df = load_dataset(train_data_path, balanced_size=balanced_train_size)
    val_df = load_dataset(val_data_path, balanced_size=balanced_val_size)
    test_df = load_dataset(test_data_path, balanced_size=balanced_test_size)

    # Balance readmission rates between templates in training set if requested
    if balance_readmission_rates:
        print("\nBalancing template readmission rates in training set...")
        train_df = balance_template_readmission_rates(train_df, label_column=label_column)
    
    # Extract texts and labels
    train_texts = train_df[text_column].tolist()
    train_labels = train_df[label_column].tolist()
    
    val_texts = val_df[text_column].tolist()
    val_labels = val_df[label_column].tolist()
    
    test_texts = test_df[text_column].tolist()
    test_labels = test_df[label_column].tolist()
    
    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"{'='*60}\n")
    
    # Log dataset sizes to W&B
    if wandb_project:
        wandb.config.update({
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "test_size": len(test_texts),
            "train_readmission_rate": np.mean(train_labels),
            "val_readmission_rate": np.mean(val_labels),
            "test_readmission_rate": np.mean(test_labels),
        })
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    except AttributeError:
        print("Fast tokenizer failed, falling back to slow tokenizer...")
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side="right")
        
    # Make sure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Define preprocess function
    def preprocess_function(examples):
        if max_seq is None:
            # No truncation - work with full notes
            return tokenizer(examples, truncation=False)
        else:
            # Truncate to max_seq length
            return tokenizer(examples, truncation=True, max_length=max_seq)

    # Create train, validation, and test datasets
    from datasets import Dataset
    
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda examples: preprocess_function(examples["text"]), 
        batched=True
    )
    tokenized_val = val_dataset.map(
        lambda examples: preprocess_function(examples["text"]), 
        batched=True
    )
    tokenized_test = test_dataset.map(
        lambda examples: preprocess_function(examples["text"]), 
        batched=True
    )

    # Prepare data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    print(f"\nLoading model: {model_name}")

    # Load base model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification",
        pad_token_id=tokenizer.pad_token_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')

    print("Model loaded successfully!")
    
    # Configure PEFT (LoRA)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Create PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if gradient_checkpointing:
        # Recompute activations in the backward pass instead of storing them: large
        # activation-memory saving (~40GB A100 OOMs on the longest full notes without
        # it) for roughly 30% slower steps. The DATA the model sees is unchanged, so
        # runs with and without this stay comparable — unlike --max-seq, which
        # truncates notes.
        #
        # With LoRA the base weights are frozen, so nothing in the checkpointed
        # segment requires grad and the backward pass would break; enabling input
        # grads on the embeddings fixes that. use_reentrant=False is the supported
        # implementation and is what handles frozen params correctly.
        model.enable_input_require_grads()
        print("Gradient checkpointing ON (activations recomputed; ~30% slower, big memory saving)")

    # Load accuracy metric
    accuracy = evaluate.load("accuracy")

    # Define compute_metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # predictions are logits (numpy array)
        # Convert to probabilities using numpy for memory efficiency
        exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        pred_probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        pred_probs = pred_probs[:, 1]  # Get probability of class 1
        
        # Get predicted classes
        pred_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        acc = accuracy.compute(predictions=pred_classes, references=labels)
        f1 = f1_score(labels, pred_classes)
        auc = roc_auc_score(labels, pred_probs)
        
        return {
            'accuracy': acc['accuracy'],
            'f1': f1,
            'auc': auc
        }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define training arguments
    # Keep the EFFECTIVE batch = batch_size (so LR/optimization is unchanged, AUC stable):
    # split it into per_device_batch_size run in parallel x gradient accumulation steps.
    grad_accum = max(1, batch_size // max(1, per_device_batch_size))
    print(f"Batch: effective={batch_size} = per_device={per_device_batch_size} x grad_accum={grad_accum}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        report_to="wandb" if wandb_project else "none",
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        dataloader_num_workers=4,           # prefetch data
        dataloader_pin_memory=True,         # faster CPU->GPU transfer
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
    )
    
    # Add early stopping callback to prevent overfitting
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0.001  # Minimum improvement threshold
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    # Train the model, resuming from the last checkpoint if one exists
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        print(f"\nResuming training from checkpoint: {last_checkpoint}")
    else:
        print("\nStarting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save the model
    print(f"\nSaving model to {final_model_dir}")
    trainer.save_model(final_model_dir)
    print(f"Model saved successfully to {final_model_dir}")

    # Keep only final_model/: load_best_model_at_end already restored the best
    # checkpoint before save_model, so the rolling checkpoint-* dirs are now
    # redundant. Remove them so the sweep persists just the final models.
    import glob, shutil
    for ckpt in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        shutil.rmtree(ckpt, ignore_errors=True)
        print(f"Removed intermediate checkpoint: {ckpt}")

    # Evaluate on test set
    print("\nEvaluating model on test set...")
    predictions = trainer.predict(tokenized_test)

    # Get predictions and labels
    pred_probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
    pred_classes = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # Calculate overall metrics
    auc_score = roc_auc_score(true_labels, pred_probs)
    f1_score_val = f1_score(true_labels, pred_classes)
    accuracy_score_val = accuracy_score(true_labels, pred_classes)

    print(f"\n{'='*60}")
    print(f"Test Set Results - Overall")
    print(f"{'='*60}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"F1 Score: {f1_score_val:.4f}")
    print(f"Accuracy: {accuracy_score_val:.4f}")
    print(f"{'='*60}\n")

    # Save predictions and labels for later analysis
    predictions_data = {
        'true_labels': true_labels,
        'pred_probs': pred_probs,
        'pred_classes': pred_classes
    }

    # Add template information if available
    if 'template' in test_df.columns:
        predictions_data['template'] = test_df['template'].values

    # Save as numpy file for easy reloading
    predictions_path = os.path.join(output_dir, 'predictions.npz')
    np.savez(predictions_path, **predictions_data)
    print(f"Predictions saved to {predictions_path}")

    # Calculate metrics separately for template A and B if template column exists
    template_metrics = {}
    if 'template' in test_df.columns:
        print(f"\n{'='*60}")
        print(f"Test Set Results - By Template")
        print(f"{'='*60}")

        for template in ['A', 'B']:
            # Get indices for this template
            template_mask = test_df['template'].values == template

            if template_mask.sum() > 0:
                # Extract predictions and labels for this template
                template_pred_probs = pred_probs[template_mask]
                template_pred_classes = pred_classes[template_mask]
                template_true_labels = true_labels[template_mask]

                # Calculate metrics (AUC is undefined when a template has no positives)
                template_auc = safe_auc(template_true_labels, template_pred_probs)
                template_f1 = f1_score(template_true_labels, template_pred_classes)
                template_acc = accuracy_score(template_true_labels, template_pred_classes)

                # Store metrics
                template_metrics[f'template_{template}_auc'] = template_auc
                template_metrics[f'template_{template}_f1'] = template_f1
                template_metrics[f'template_{template}_accuracy'] = template_acc
                template_metrics[f'template_{template}_n_samples'] = template_mask.sum()

                print(f"\nTemplate {template} (n={template_mask.sum()}):")
                print(f"  AUC Score: {template_auc:.4f}")
                print(f"  F1 Score: {template_f1:.4f}")
                print(f"  Accuracy: {template_acc:.4f}")

        print(f"{'='*60}\n")
    
    # Plot ROC curves - Overall and by template
    if 'template' in test_df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Overall ROC curve
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        axes[0].plot(fpr, tpr, label=f'Overall (AUC = {auc_score:.4f})', linewidth=2)
        axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve - Overall')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Template A ROC curve
        template_a_mask = test_df['template'].values == 'A'
        if template_a_mask.sum() > 0:
            fpr_a, tpr_a, _ = roc_curve(true_labels[template_a_mask], pred_probs[template_a_mask])
            template_a_auc = template_metrics.get('template_A_auc', 0)
            axes[1].plot(fpr_a, tpr_a, label=f'Template A (AUC = {template_a_auc:.4f})', linewidth=2, color='blue')
            axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title(f'ROC Curve - Template A (n={template_a_mask.sum()})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Template B ROC curve
        template_b_mask = test_df['template'].values == 'B'
        if template_b_mask.sum() > 0:
            fpr_b, tpr_b, _ = roc_curve(true_labels[template_b_mask], pred_probs[template_b_mask])
            template_b_auc = template_metrics.get('template_B_auc', 0)
            axes[2].plot(fpr_b, tpr_b, label=f'Template B (AUC = {template_b_auc:.4f})', linewidth=2, color='orange')
            axes[2].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[2].set_xlabel('False Positive Rate')
            axes[2].set_ylabel('True Positive Rate')
            axes[2].set_title(f'ROC Curve - Template B (n={template_b_mask.sum()})')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'roc_curves.png')
        plt.savefig(plot_path, dpi=150)
        print(f"ROC curves saved to {plot_path}")
        plt.close()
    else:
        # Original single ROC curve plot if no template column
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Test Set')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plot_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(plot_path)
        print(f"ROC curve saved to {plot_path}")
        plt.close()
    
    # Log to W&B
    if wandb_project:
        log_dict = {
            "test/auc": auc_score,
            "test/f1": f1_score_val,
            "test/accuracy": accuracy_score_val,
            "test/roc_curve": wandb.Image(plot_path),
        }
        # Add template-specific metrics if available
        for key, value in template_metrics.items():
            log_dict[f"test/{key}"] = value
        wandb.log(log_dict)
        wandb.finish()
    
    # Save results summary
    results_summary = {
        'auc_score': auc_score,
        'f1_score': f1_score_val,
        'accuracy': accuracy_score_val,
        'train_samples': len(train_texts),
        'val_samples': len(val_texts),
        'test_samples': len(test_texts),
        'train_readmission_rate': np.mean(train_labels),
        'val_readmission_rate': np.mean(val_labels),
        'test_readmission_rate': np.mean(test_labels),
        'model_name': model_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }

    # Add template-specific metrics to results summary
    results_summary.update(template_metrics)

    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        # Write overall metrics first
        f.write("="*60 + "\n")
        f.write("Overall Test Results\n")
        f.write("="*60 + "\n")
        f.write(f"auc_score: {auc_score}\n")
        f.write(f"f1_score: {f1_score_val}\n")
        f.write(f"accuracy: {accuracy_score_val}\n")

        # Write template-specific metrics if available
        if template_metrics:
            f.write("\n" + "="*60 + "\n")
            f.write("Template-Specific Results\n")
            f.write("="*60 + "\n")
            for template in ['A', 'B']:
                auc_key = f'template_{template}_auc'
                if auc_key in template_metrics:
                    f.write(f"\nTemplate {template}:\n")
                    f.write(f"  auc: {template_metrics[f'template_{template}_auc']}\n")
                    f.write(f"  f1: {template_metrics[f'template_{template}_f1']}\n")
                    f.write(f"  accuracy: {template_metrics[f'template_{template}_accuracy']}\n")
                    f.write(f"  n_samples: {template_metrics[f'template_{template}_n_samples']}\n")

        # Write other metadata
        f.write("\n" + "="*60 + "\n")
        f.write("Metadata\n")
        f.write("="*60 + "\n")
        f.write(f"train_samples: {len(train_texts)}\n")
        f.write(f"val_samples: {len(val_texts)}\n")
        f.write(f"test_samples: {len(test_texts)}\n")
        f.write(f"train_readmission_rate: {np.mean(train_labels)}\n")
        f.write(f"val_readmission_rate: {np.mean(val_labels)}\n")
        f.write(f"test_readmission_rate: {np.mean(test_labels)}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"learning_rate: {learning_rate}\n")
    print(f"Results summary saved to {results_path}")
    
    return trainer.model, tokenizer, auc_score, results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train classifier on preprocessed data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--train-data',
        type=str,
        required=True,
        help='Path to training parquet file'
    )
    parser.add_argument(
        '--val-data',
        type=str,
        required=True,
        help='Path to validation parquet file'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test parquet file'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Name of column containing text to classify'
    )
    parser.add_argument(
        '--label-column',
        type=str,
        default='label',
        help='Name of column containing labels'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen3-0.6B',
        help='HuggingFace model name'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='EFFECTIVE batch size (per_device x grad_accum). Changing this changes '
             'optimization/AUC; keep it fixed and use --per-device-batch-size for speed.'
    )
    parser.add_argument(
        '--gradient-checkpointing',
        action='store_true',
        help='Recompute activations in the backward pass instead of storing them. ~30%% '
             'slower but a large activation-memory saving — use when full notes OOM on a '
             '40GB GPU. Unlike --max-seq it does NOT change the data the model sees, so '
             'runs with and without it remain comparable.'
    )
    parser.add_argument(
        '--per-device-batch-size',
        type=int,
        default=1,
        help='Notes run in parallel per forward pass. Raising this (memory permitting) '
             'speeds up training WITHOUT changing AUC, because grad_accum is reduced to '
             'keep the effective batch (--batch-size) constant.'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models',
        help='Directory to save models and results'
    )
    
    # LoRA arguments
    parser.add_argument(
        '--lora-r',
        type=int,
        default=16,
        help='LoRA rank'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha'
    )
    parser.add_argument(
        '--lora-dropout',
        type=float,
        default=0.1,
        help='LoRA dropout'
    )

    # Tokenization arguments
    parser.add_argument(
        '--max-seq',
        type=int,
        default=None,
        help='Maximum sequence length for truncation (None = no truncation)'
    )

    # Balanced sampling arguments
    parser.add_argument(
        '--balanced-train-size',
        type=int,
        default=None,
        help='Balance training set to this size (half template A, half template B)'
    )
    parser.add_argument(
        '--balanced-val-size',
        type=int,
        default=None,
        help='Balance validation set to this size (half template A, half template B)'
    )
    parser.add_argument(
        '--balanced-test-size',
        type=int,
        default=None,
        help='Balance test set to this size (half template A, half template B)'
    )

    # Early stopping
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=2,
        help='Number of epochs with no improvement before stopping'
    )

    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    # Readmission rate balancing
    parser.add_argument(
        '--balance-readmission-rates',
        action='store_true',
        default=False,
        help='Equalize template B readmission rate to match template A in the training split'
    )

    # W&B arguments
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='W&B project name (optional)'
    )
    parser.add_argument(
        '--wandb-entity',
        type=str,
        default=None,
        help='W&B entity/username (optional)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='W&B run name (optional)'
    )
    
    args = parser.parse_args()
    
    # Train model
    model, tokenizer, auc_score, results = train_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        test_data_path=args.test_data,
        text_column=args.text_column,
        label_column=args.label_column,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        per_device_batch_size=args.per_device_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_seq=args.max_seq,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
        balanced_train_size=args.balanced_train_size,
        balanced_val_size=args.balanced_val_size,
        balanced_test_size=args.balanced_test_size,
        early_stopping_patience=args.early_stopping_patience,
        balance_readmission_rates=args.balance_readmission_rates,
        seed=args.seed
    )
    
    if results is not None:
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"F1 Score: {results.get('f1_score', 'N/A'):.4f}")
        print(f"Accuracy: {results.get('accuracy', 'N/A'):.4f}")
        print(f"{'='*60}\n")

