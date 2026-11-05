#!/usr/bin/env python3
"""
Data Processing Script for Flexible Fine-tuning
This script processes any dataset and prepares it for training.
It assumes notes are already in either template A or template B (not both).
"""

import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import random

def load_data(path):
    """Load a local Parquet (default) or CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local file not found: {path}")
    print(f"Loading from local file: {path}")
    if str(path).lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows.")
    return df


def create_biased_dataset(
    df_template_a, 
    df_template_b, 
    p_train=0.5, 
    label_column='readmit_binary',
    text_column='full_note',
    target_size=None, 
    max_samples_per_template=None, 
    random_seed=42
):
    """
    Create a dataset with controlled template-outcome correlation using p-parameter.
    
    Strategy:
    1. Determine total readmissions and non-readmissions needed
    2. Allocate readmissions: p_train proportion from Template A, (1-p_train) from Template B
    3. Allocate non-readmissions to ensure equal template sizes
    4. If max_samples_per_template is specified, both templates will have exactly that many samples
    
    Args:
        df_template_a: DataFrame with template A notes (must have label_column)
        df_template_b: DataFrame with template B notes (must have label_column)
        p_train: Proportion of readmission cases from Template A (0.0 to 1.0)
                 0.0 = all readmissions from Template B (maximum bias toward B)
                 0.5 = no correlation (control, balanced)
                 1.0 = all readmissions from Template A (maximum bias toward A)
        label_column: Name of column containing labels
        text_column: Name of column containing text
        target_size: Total number of samples desired (if None, uses all available)
        max_samples_per_template: Maximum number of samples per template
                                 If specified, both templates will have exactly that many samples
        random_seed: Random seed for reproducibility
    
    Returns:
        Combined DataFrame with specified correlation and template column
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Filter out NaN values in label column
    df_template_a = df_template_a.dropna(subset=[label_column])
    df_template_b = df_template_b.dropna(subset=[label_column])
    
    # Separate positives and negatives for each template
    positives_a = df_template_a[df_template_a[label_column] == 1]
    negatives_a = df_template_a[df_template_a[label_column] == 0]
    positives_b = df_template_b[df_template_b[label_column] == 1]
    negatives_b = df_template_b[df_template_b[label_column] == 0]
    
    available_pos_a = len(positives_a)
    available_neg_a = len(negatives_a)
    available_pos_b = len(positives_b)
    available_neg_b = len(negatives_b)
    
    print(f"\n{'='*60}")
    print(f"Creating dataset with p_train: {p_train:.4f}")
    print(f"{'='*60}")
    print(f"Available - Template A: {available_pos_a} positives, {available_neg_a} negatives")
    print(f"Available - Template B: {available_pos_b} positives, {available_neg_b} negatives")
    
    # Determine total readmissions and non-readmissions needed
    total_available_cases = available_pos_a + available_neg_a + available_pos_b + available_neg_b
    total_available_readmits = available_pos_a + available_pos_b
    base_readmission_rate = total_available_readmits / total_available_cases if total_available_cases > 0 else 0
    
    if max_samples_per_template is not None:
        # If max_samples_per_template is specified, we want both templates to have that many samples
        total_samples_target = 2 * max_samples_per_template
        total_readmissions_needed = int(total_samples_target * base_readmission_rate)
        total_non_readmissions_needed = total_samples_target - total_readmissions_needed
        print(f"Using max_samples_per_template={max_samples_per_template}:")
        print(f"  Target total samples: {total_samples_target}")
        print(f"  Target readmissions: {total_readmissions_needed}")
        print(f"  Target non-readmissions: {total_non_readmissions_needed}")
    elif target_size is not None:
        # Use specified target_size
        total_readmissions_needed = int(target_size * base_readmission_rate)
        total_non_readmissions_needed = target_size - total_readmissions_needed
        print(f"Using specified target_size={target_size}:")
        print(f"  Target readmissions: {total_readmissions_needed}")
        print(f"  Target non-readmissions: {total_non_readmissions_needed}")
    else:
        # Use all available data
        total_readmissions_needed = total_available_readmits
        total_non_readmissions_needed = total_available_cases - total_readmissions_needed
        print(f"Using all available data:")
        print(f"  Total readmissions: {total_readmissions_needed}")
        print(f"  Total non-readmissions: {total_non_readmissions_needed}")
    
    # Allocate readmissions: p_train proportion from Template A, (1-p_train) from Template B
    readmissions_from_a = int(total_readmissions_needed * p_train)
    readmissions_from_b = total_readmissions_needed - readmissions_from_a
    
    # Allocate non-readmissions to ensure equal template sizes
    if max_samples_per_template is not None:
        # When max_samples_per_template is specified, we want both templates to have exactly that many samples
        non_readmissions_from_a = max_samples_per_template - readmissions_from_a
        non_readmissions_from_b = max_samples_per_template - readmissions_from_b
    else:
        # For other cases, calculate to balance template sizes
        diff = readmissions_from_b - readmissions_from_a
        non_readmissions_from_a = int((total_non_readmissions_needed + diff) / 2)
        non_readmissions_from_b = total_non_readmissions_needed - non_readmissions_from_a
    
    print(f"\nInitial sampling plan (before capping):")
    print(f"  Template A: {readmissions_from_a} readmissions, {non_readmissions_from_a} non-readmissions (total: {readmissions_from_a + non_readmissions_from_a})")
    print(f"  Template B: {readmissions_from_b} readmissions, {non_readmissions_from_b} non-readmissions (total: {readmissions_from_b + non_readmissions_from_b})")
    
    # Cap at available data
    readmissions_from_a = min(readmissions_from_a, available_pos_a)
    readmissions_from_b = min(readmissions_from_b, available_pos_b)
    non_readmissions_from_a = min(non_readmissions_from_a, available_neg_a)
    non_readmissions_from_b = min(non_readmissions_from_b, available_neg_b)
    
    # Sample from each template
    sampled_pos_a = positives_a.sample(n=readmissions_from_a, random_state=random_seed, replace=False)
    sampled_neg_a = negatives_a.sample(n=non_readmissions_from_a, random_state=random_seed, replace=False)
    sampled_a = pd.concat([sampled_pos_a, sampled_neg_a], ignore_index=True)
    
    sampled_pos_b = positives_b.sample(n=readmissions_from_b, random_state=random_seed, replace=False)
    sampled_neg_b = negatives_b.sample(n=non_readmissions_from_b, random_state=random_seed, replace=False)
    sampled_b = pd.concat([sampled_pos_b, sampled_neg_b], ignore_index=True)
    
    # Add template identifier
    sampled_a['template'] = 'A'
    sampled_b['template'] = 'B'
    
    # Combine and shuffle
    combined_df = pd.concat([sampled_a, sampled_b], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Verify achieved correlation and readmission rate
    actual_rate_a = sampled_a[label_column].mean()
    actual_rate_b = sampled_b[label_column].mean()
    actual_p = sampled_a[label_column].sum() / (sampled_a[label_column].sum() + sampled_b[label_column].sum()) if (sampled_a[label_column].sum() + sampled_b[label_column].sum()) > 0 else 0.5
    overall_rate = combined_df[label_column].mean()
    
    print(f"\nAchieved results:")
    print(f"  Template A: {actual_rate_a:.4f} readmission rate (n={len(sampled_a)}, {sampled_a[label_column].sum()} readmissions)")
    print(f"  Template B: {actual_rate_b:.4f} readmission rate (n={len(sampled_b)}, {sampled_b[label_column].sum()} readmissions)")
    print(f"  Overall readmission rate: {overall_rate:.4f} (base rate: {base_readmission_rate:.4f})")
    print(f"  Actual p_train: {actual_p:.4f} (proportion of readmissions from Template A, target: {p_train:.4f})")
    print(f"  Total samples: {len(combined_df)} (Template A: {len(sampled_a)}, Template B: {len(sampled_b)})")
    print(f"{'='*60}\n")
    
    return combined_df


def _assign_paired_train(df_paired, p_train, label_column, max_samples_per_template, random_seed,
                         name="Train"):
    """
    Build a biased split from paired notes at correlation strength p_train.

    Each positive note is assigned to Template A with probability p_train,
    Template B with probability (1-p_train). Negatives are distributed to
    equalise template sizes. Each note_id appears at most once (either as A
    or B), so no clinical content is duplicated within the split.

    Used for the training split and — when p_eval is set — for the val/test
    splits too, so the per-template readmission rate matches train-vs-test.
    """
    rng = np.random.RandomState(random_seed)

    positives = df_paired[df_paired[label_column] == 1].sample(frac=1, random_state=random_seed).reset_index(drop=True)
    negatives = df_paired[df_paired[label_column] == 0].sample(frac=1, random_state=random_seed).reset_index(drop=True)

    n_pos = len(positives)
    n_pos_a = int(round(n_pos * p_train))
    n_pos_b = n_pos - n_pos_a

    pos_a = positives.iloc[:n_pos_a].copy()
    pos_b = positives.iloc[n_pos_a:].copy()

    if max_samples_per_template is not None:
        # How many negatives each template WANTS to reach max_samples.
        want_neg_a = max(0, max_samples_per_template - n_pos_a)
        want_neg_b = max(0, max_samples_per_template - n_pos_b)
        total_want = want_neg_a + want_neg_b
        avail = len(negatives)
        if total_want <= avail:
            n_neg_a, n_neg_b = want_neg_a, want_neg_b
        else:
            # Not enough negatives: split proportionally so NEITHER template is starved.
            # (The old greedy 'fill A first' zeroed out B when max_samples exceeded supply,
            # creating a spurious template<->label correlation.)
            n_neg_a = int(round(avail * want_neg_a / total_want)) if total_want else 0
            n_neg_b = avail - n_neg_a
    else:
        # Solve for equal template sizes: n_neg_a - n_neg_b = n_pos_b - n_pos_a
        # using all available negatives: n_neg_a + n_neg_b = len(negatives)
        n_neg_a = max(0, min((len(negatives) + n_pos_b - n_pos_a) // 2, len(negatives)))
        n_neg_b = max(0, min(len(negatives) - n_neg_a, len(negatives)))

    neg_a = negatives.iloc[:n_neg_a].copy()
    neg_b = negatives.iloc[n_neg_a:n_neg_a + n_neg_b].copy()

    rows = []
    for subset, template in [(pos_a, 'A'), (neg_a, 'A'), (pos_b, 'B'), (neg_b, 'B')]:
        sub = subset.copy()
        sub['text'] = sub['text_a' if template == 'A' else 'text_b']
        sub['template'] = template
        rows.append(sub[['text', label_column, 'template']])

    result = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    actual_p = pos_a.shape[0] / max(n_pos, 1)
    rate_a = result[result['template'] == 'A'][label_column].mean()
    rate_b = result[result['template'] == 'B'][label_column].mean()
    print(f"  {name}: {len(result)} samples | p achieved={actual_p:.3f} | "
          f"A n={sum(result['template']=='A')} readmit={rate_a:.3f} | "
          f"B n={sum(result['template']=='B')} readmit={rate_b:.3f}")
    return result


def _assign_paired_eval(df_paired, label_column, random_seed):
    """
    Build a balanced eval split from paired notes.

    Each note_id appears exactly once — half assigned to Template A,
    half to Template B, stratified by label. No content leakage.
    """
    rows = []
    for label_val in [0, 1]:
        subset = df_paired[df_paired[label_column] == label_val].sample(frac=1, random_state=random_seed).reset_index(drop=True)
        mid = len(subset) // 2
        for i, (part, template) in enumerate([(subset.iloc[:mid], 'A'), (subset.iloc[mid:], 'B')]):
            sub = part.copy()
            sub['text'] = sub['text_a' if template == 'A' else 'text_b']
            sub['template'] = template
            rows.append(sub[['text', label_column, 'template']])
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)


def attach_group_column(df, group_column, group_map_path, group_map_key):
    """
    Join a grouping column (e.g. pat_id) onto df from a separate mapping file.

    Needed because the note-level parquets carry only pat_enc_csn_id / note_id;
    the patient key lives upstream (e.g. filtered_notes.parquet).
    """
    if group_column in df.columns:
        return df

    if group_map_path is None:
        raise ValueError(
            f"Group column '{group_column}' not found in the input data "
            f"(columns: {list(df.columns)}). Supply --group-map-path pointing at a "
            f"file that contains both '{group_map_key}' and '{group_column}'."
        )

    lower = group_map_path.lower()
    if lower.endswith(('.pkl', '.pickle')):
        gmap = pd.read_pickle(group_map_path)
    elif lower.endswith('.parquet'):
        # Read only the join key and group column. These mapping files also carry
        # note text; there is no reason to pull it into memory to build an ID map.
        gmap = pd.read_parquet(group_map_path, columns=[group_map_key, group_column])
    else:
        gmap = load_data(group_map_path)

    for col in (group_map_key, group_column):
        if col not in gmap.columns:
            raise ValueError(f"Group map is missing column '{col}'. Has: {list(gmap.columns)}")
    if group_map_key not in df.columns:
        raise ValueError(
            f"Join key '{group_map_key}' not found in the input data (columns: {list(df.columns)})"
        )

    gmap = gmap[[group_map_key, group_column]].drop_duplicates(subset=[group_map_key])
    before = len(df)
    df = df.merge(gmap, on=group_map_key, how='left')
    n_missing = df[group_column].isna().sum()
    print(f"  Joined '{group_column}' from {group_map_path} on '{group_map_key}': "
          f"{before - n_missing}/{before} rows matched")
    if n_missing:
        raise ValueError(
            f"{n_missing} rows have no '{group_column}' after the join. Grouped splitting "
            f"would silently treat them as one group — fix the mapping file first."
        )
    return df


def split_three_way(df, label_column, val_ratio, test_ratio, random_seed,
                    group_column=None, stratify_groups=False, template_column=None,
                    match_rates=False, n_restarts=8):
    """
    Split df into train/val/test.

    - group_column is None: row-level split stratified by label (original behavior).
    - group_column set, match_rates True: balanced_group_split — solves the group
      assignment as a constrained optimisation so per-template label rates, split
      ratios and case mix all match across splits. RECOMMENDED for shortcut studies;
      on the Dec9 pair it holds the per-template rate to 0.07pp vs 3.2-3.8pp for the
      two options below, i.e. below the 2.8pp template gap being studied rather than
      above it. See balanced_group_split.py.
    - group_column set, stratify_groups True: StratifiedGroupKFold on template x label.
      Keeps groups intact and is better than nothing, but it balances folds greedily one
      at a time rather than optimising the three splits jointly, so in practice it barely
      beats GroupShuffleSplit on per-template rate. Ratios also become approximate.
    - group_column set, neither flag: GroupShuffleSplit — every row sharing a group value
      lands in exactly one split. Exact ratios, but label balance drifts.
    """
    val_test_ratio = val_ratio + test_ratio
    val_share_of_temp = 1 - (val_ratio / val_test_ratio)

    if group_column is None:
        train_df, temp_df = train_test_split(
            df, test_size=val_test_ratio, random_state=random_seed, stratify=df[label_column]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=val_share_of_temp, random_state=random_seed,
            stratify=temp_df[label_column]
        )
        return train_df, val_df, test_df

    groups = df[group_column]
    print(f"\nGrouped split on '{group_column}': {groups.nunique()} groups across {len(df)} rows "
          f"({len(df) / max(groups.nunique(), 1):.2f} rows/group)")

    def _strat_key(frame):
        # Stratify on label, combined with template when available, so each
        # (template, label) cell is balanced across splits.
        y = frame[label_column].astype(str)
        if template_column and template_column in frame.columns:
            y = y + '|' + frame[template_column].astype(str)
        return y

    if match_rates:
        # Works both as a script (python finetuning/process_data.py) and as a
        # module (python -m finetuning.process_data, which is how sweep_p_train
        # invokes it — only the repo root is on sys.path there).
        try:
            from balanced_group_split import balanced_group_split, feasibility_report
        except ModuleNotFoundError:
            from finetuning.balanced_group_split import (
                balanced_group_split, feasibility_report)
        train_ratio = 1.0 - val_ratio - test_ratio
        print(feasibility_report(df, label_column, group_column, template_column,
                                 (train_ratio, val_ratio, test_ratio)))
        train_df, val_df, test_df = balanced_group_split(
            df, label_column, group_column, template_column,
            ratios=(train_ratio, val_ratio, test_ratio),
            random_seed=random_seed, n_restarts=n_restarts, verbose=True)
        return train_df, val_df, test_df

    used_stratified = False
    if stratify_groups:
        from sklearn.model_selection import StratifiedGroupKFold
        # test is ~1/k_test of the data; val is ~1/k_val of the remainder.
        # For the default 64/16/20 this gives k_test=5, k_val=5 -> 20/16/64 exactly in expectation.
        k_test = max(2, round(1 / test_ratio))
        k_val = max(2, round((1 - test_ratio) / val_ratio))
        try:
            sgkf1 = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=random_seed)
            rest_idx, test_idx = next(sgkf1.split(df, _strat_key(df), groups))
            test_df, rest_df = df.iloc[test_idx], df.iloc[rest_idx]

            sgkf2 = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=random_seed)
            train_idx, val_idx = next(sgkf2.split(rest_df, _strat_key(rest_df), rest_df[group_column]))
            train_df, val_df = rest_df.iloc[train_idx], rest_df.iloc[val_idx]
            used_stratified = True
            strat_desc = f"{template_column} x {label_column}" if (template_column and template_column in df.columns) else label_column
            print(f"  Stratified-group split on ({strat_desc}); ratios are approximate "
                  f"(k_test={k_test}, k_val={k_val})")
        except ValueError as e:
            print(f"  StratifiedGroupKFold failed ({e}); falling back to GroupShuffleSplit")

    if not used_stratified:
        from sklearn.model_selection import GroupShuffleSplit
        gss1 = GroupShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=random_seed)
        train_idx, temp_idx = next(gss1.split(df, groups=groups))
        train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_share_of_temp, random_state=random_seed)
        val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_column]))
        val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]

    # Fail loudly rather than silently shipping a leaky split.
    tr_g, va_g, te_g = set(train_df[group_column]), set(val_df[group_column]), set(test_df[group_column])
    for a_name, a, b_name, b in [('train', tr_g, 'val', va_g), ('train', tr_g, 'test', te_g),
                                 ('val', va_g, 'test', te_g)]:
        overlap = a & b
        if overlap:
            raise AssertionError(f"{len(overlap)} '{group_column}' values appear in both {a_name} and {b_name}")
    print(f"  Verified: no '{group_column}' value appears in more than one split")
    print(f"  Split sizes — train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    print(f"  Label rate — train: {train_df[label_column].mean():.4f} | "
          f"val: {val_df[label_column].mean():.4f} | test: {test_df[label_column].mean():.4f} "
          f"(overall {df[label_column].mean():.4f})")

    return train_df, val_df, test_df


def process_dataset(
    input_path=None,
    template_a_path=None,
    template_b_path=None,
    text_column='full_note',
    label_column='readmit_binary',
    template_column='template',
    output_dir='./processed_data',
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    p_train=None,
    target_size=None,
    max_samples_per_template=None,
    group_column=None,
    group_map_path=None,
    group_map_key='pat_enc_csn_id',
    stratify_groups=False,
    match_rates=False,
    p_eval=None
):
    """
    Process a dataset and split it into train/val/test sets.
    
    Can operate in two modes:
    1. Simple mode: Process a single dataset (provide input_path)
    2. Bias creation mode: Create biased dataset from two templates (provide template_a_path and template_b_path)
    
    Args:
        input_path: Path to input Parquet file - for simple mode
        template_a_path: Path to template A Parquet file - for bias creation mode
        template_b_path: Path to template B Parquet file - for bias creation mode
        text_column: Name of column containing text
        label_column: Name of column containing labels
        template_column: Name of column containing template identifier (A or B)
        output_dir: Directory to save processed datasets
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        p_train: Bias parameter (0.0 to 1.0) - proportion of readmissions from Template A
                 If provided, will create biased dataset from template_a_path and template_b_path
        target_size: Total number of samples desired (for bias creation mode)
        max_samples_per_template: Maximum samples per template (for bias creation mode)
        group_column: If set (e.g. 'pat_id'), split by group so all rows sharing a value
                      land in one split. Prevents patient-level leakage when one patient
                      contributed multiple encounters. Default None = row-level split.
        group_map_path: File to join group_column from, if it isn't already a column
        group_map_key: Join key linking the input data to group_map_path
    
    Returns:
        Paths to saved train, val, and test Parquet files
    """
    # Determine mode: bias creation or simple processing
    if p_train is not None:
        if template_a_path is None or template_b_path is None:
            raise ValueError("For bias creation mode (p_train provided), both template_a_path and template_b_path must be provided")

        eval_desc = (f"val/test biased at p_eval={p_eval} (per-template rate matched to train)"
                     if p_eval is not None else "val/test balanced at p=0.5")
        print(f"\n{'='*60}")
        print(f"PAIRED BIAS CREATION MODE")
        print(f"p_train={p_train} (train); {eval_desc}")
        print(f"{'='*60}")

        df_a = load_data(template_a_path)
        df_b = load_data(template_b_path)

        if 'note_id' not in df_a.columns or 'note_id' not in df_b.columns:
            raise ValueError("Paired mode requires a 'note_id' column in both template files. "
                             "Re-run make_synthetic_templates.py to regenerate them.")

        # Merge into paired dataframe: one row per note, text_a and text_b columns
        df_a_renamed = df_a.rename(columns={text_column: 'text_a'})
        df_b_renamed = df_b[[text_column, 'note_id']].rename(columns={text_column: 'text_b'})
        paired = df_a_renamed.merge(df_b_renamed, on='note_id').dropna(subset=[label_column])
        paired[label_column] = paired[label_column].astype(int)
        print(f"  Paired notes: {len(paired)} | positives: {paired[label_column].sum()} | "
              f"negatives: {(paired[label_column]==0).sum()}")

        # Split note_ids into train / val / test (no note appears in multiple splits).
        # With group_column set, no patient appears in multiple splits either.
        if group_column:
            paired = attach_group_column(paired, group_column, group_map_path, group_map_key)
        # Paired mode assigns template after splitting, so no template column exists yet;
        # stratify on label only.
        train_paired, val_paired, test_paired = split_three_way(
            paired, label_column, val_ratio, test_ratio, random_seed, group_column,
            stratify_groups=stratify_groups, match_rates=match_rates, template_column=None
        )

        print(f"\nCreating biased train split (p_train={p_train}) ...")
        train_df = _assign_paired_train(train_paired, p_train, label_column, max_samples_per_template,
                                        random_seed, name="Train")

        if p_eval is not None:
            # Bias val/test at p_eval so each template's readmission rate matches train.
            # max_samples_per_template=None -> use all eval notes (no capping).
            print(f"\nCreating biased val/test splits (p_eval={p_eval}) ...")
            val_df = _assign_paired_train(val_paired, p_eval, label_column, None, random_seed, name="Val")
            test_df = _assign_paired_train(test_paired, p_eval, label_column, None, random_seed, name="Test")
        else:
            print(f"\nCreating balanced val/test splits ...")
            val_df = _assign_paired_eval(val_paired, label_column, random_seed)
            test_df = _assign_paired_eval(test_paired, label_column, random_seed)

        # Rename label column to standard 'label'
        train_df = train_df.rename(columns={label_column: 'label'})
        val_df = val_df.rename(columns={label_column: 'label'})
        test_df = test_df.rename(columns={label_column: 'label'})

        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, 'train.parquet')
        val_path = os.path.join(output_dir, 'val.parquet')
        test_path = os.path.join(output_dir, 'test.parquet')

        train_df[['text', 'label', 'template']].to_parquet(train_path, index=False)
        val_df[['text', 'label', 'template']].to_parquet(val_path, index=False)
        test_df[['text', 'label', 'template']].to_parquet(test_path, index=False)

        print(f"\nSaved datasets to:")
        print(f"  Train: {train_path} ({len(train_df)} samples)")
        print(f"  Validation: {val_path} ({len(val_df)} samples)")
        print(f"  Test: {test_path} ({len(test_df)} samples)")
        return train_path, val_path, test_path

    else:
        # Simple mode: load single dataset
        if input_path is None:
            raise ValueError("Either input_path (for simple mode) or template_a_path/template_b_path with p_train (for bias mode) must be provided")
        
        # Load data
        df = load_data(input_path)
    
    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    
    # Check required columns
    required_cols = [text_column, label_column]
    if template_column and template_column in df.columns:
        required_cols.append(template_column)
        print(f"Template distribution:")
        if template_column in df.columns:
            print(df[template_column].value_counts().to_dict())
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out NaN values
    df = df.dropna(subset=[text_column, label_column])
    print(f"After filtering NaN values: {len(df)} samples")
    
    # Convert labels to int if needed
    if df[label_column].dtype != 'int64':
        df[label_column] = df[label_column].astype(int)
    
    # Print label distribution
    print(f"\nLabel distribution:")
    label_counts = df[label_column].value_counts().to_dict()
    print(f"  Class 0: {label_counts.get(0, 0)}")
    print(f"  Class 1: {label_counts.get(1, 0)}")
    print(f"  Readmission rate: {df[label_column].mean():.4f}")
    
    # Verify split ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Split into train, validation, and test sets.
    # group_column (e.g. 'pat_id') keeps every row from one patient in a single split.
    if group_column:
        df = attach_group_column(df, group_column, group_map_path, group_map_key)
    train_df, val_df, test_df = split_three_way(
        df, label_column, val_ratio, test_ratio, random_seed, group_column,
        stratify_groups=stratify_groups, match_rates=match_rates,
        template_column=template_column
    )
    
    print(f"\n{'='*60}")
    print(f"Split Results")
    print(f"{'='*60}")
    print(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Readmission rate: {train_df[label_column].mean():.4f}")
    print(f"Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Readmission rate: {val_df[label_column].mean():.4f}")
    print(f"Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Readmission rate: {test_df[label_column].mean():.4f}")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Rename columns to standard names for training script
    # The training script expects 'text' and 'label' columns
    column_mapping = {
        text_column: 'text',
        label_column: 'label'
    }
    
    train_df_renamed = train_df.rename(columns=column_mapping)
    val_df_renamed = val_df.rename(columns=column_mapping)
    test_df_renamed = test_df.rename(columns=column_mapping)
    
    # Select only the columns we need (text, label, and optionally template)
    output_cols = ['text', 'label']
    if template_column and template_column in train_df.columns:
        output_cols.append(template_column)
        # Keep template column with original name if it's not being renamed
        if template_column not in column_mapping:
            train_df_renamed[template_column] = train_df[template_column]
            val_df_renamed[template_column] = val_df[template_column]
            test_df_renamed[template_column] = test_df[template_column]
    
    # Save datasets
    train_path = os.path.join(output_dir, 'train.parquet')
    val_path = os.path.join(output_dir, 'val.parquet')
    test_path = os.path.join(output_dir, 'test.parquet')

    train_df_renamed[output_cols].to_parquet(train_path, index=False)
    val_df_renamed[output_cols].to_parquet(val_path, index=False)
    test_df_renamed[output_cols].to_parquet(test_path, index=False)
    
    print(f"Saved datasets to:")
    print(f"  Train: {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  Test: {test_path}")
    
    return train_path, val_path, test_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process dataset and split into train/val/test sets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments - Simple mode
    parser.add_argument(
        '--input-path',
        type=str,
        default=None,
        help='Path to input Parquet file - for simple mode'
    )

    # Input arguments - Bias creation mode
    parser.add_argument(
        '--template-a-path',
        type=str,
        default=None,
        help='Path to template A Parquet file (for bias creation mode)'
    )
    parser.add_argument(
        '--template-b-path',
        type=str,
        default=None,
        help='Path to template B Parquet file (for bias creation mode)'
    )
    parser.add_argument(
        '--p-train',
        type=float,
        default=None,
        help='Bias parameter (0.0 to 1.0): proportion of readmissions from Template A. '
             'If provided, creates biased dataset from template A and B files. '
             '0.0 = all readmissions from Template B, 0.5 = balanced, 1.0 = all from Template A'
    )
    parser.add_argument(
        '--p-eval',
        type=float,
        default=None,
        help='If set, bias val/test at this p too (instead of balanced p=0.5), so each '
             'template\'s readmission rate matches train-vs-test. Set equal to --p-train for a '
             'persistent shortcut; set to 1-p_train for a counter-biased (robustness) test. '
             'Pair with --stratify-groups so the overall rate is also constant across splits.'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=None,
        help='Total number of samples desired (for bias creation mode)'
    )
    parser.add_argument(
        '--max-samples-per-template',
        type=int,
        default=None,
        help='Maximum number of samples per template (for bias creation mode). '
             'If specified, both templates will have exactly this many samples.'
    )
    
    # Column arguments
    parser.add_argument(
        '--text-column',
        type=str,
        default='full_note',
        help='Name of column containing text to classify'
    )
    parser.add_argument(
        '--label-column',
        type=str,
        default='readmit_binary',
        help='Name of column containing labels'
    )
    parser.add_argument(
        '--template-column',
        type=str,
        default='template',
        help='Name of column containing template identifier (optional)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./processed_data',
        help='Directory to save processed datasets'
    )
    
    # Split arguments
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.70,
        help='Proportion for training set'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Proportion for validation set'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Proportion for test set'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--group-column',
        type=str,
        default=None,
        help="Split by this column instead of by row (e.g. 'pat_id'). All rows sharing a "
             "value stay in one split, preventing patient-level leakage when one patient "
             "contributed multiple encounters. Grouping precludes exact label stratification."
    )
    parser.add_argument(
        '--group-map-path',
        type=str,
        default=None,
        help='File (parquet/csv/pkl) to join --group-column from, when the input data does '
             'not already carry it (e.g. filtered_notes.parquet for pat_id)'
    )
    parser.add_argument(
        '--group-map-key',
        type=str,
        default='pat_enc_csn_id',
        help='Join key linking the input data to --group-map-path'
    )
    parser.add_argument(
        '--stratify-groups',
        action='store_true',
        help='Only with --group-column: use StratifiedGroupKFold so each (template x label) '
             'cell stays balanced across splits (keeps the per-template readmission gap matched '
             'train-vs-test). Split ratios become approximate. Default off = GroupShuffleSplit.'
    )

    parser.add_argument(
        '--match-rates',
        action='store_true',
        help='Only with --group-column: solve the group assignment so per-template label '
             'rates, split ratios and case mix all match across train/val/test '
             '(balanced_group_split.py). Strictly better than --stratify-groups for '
             'shortcut studies; takes a few seconds. Overrides --stratify-groups.'
    )

    args = parser.parse_args()
    
    # Validate arguments
    if args.p_train is not None and (args.template_a_path is None or args.template_b_path is None):
        parser.error("--p-train requires both --template-a-path and --template-b-path")
    if args.p_train is None and args.input_path is None:
        parser.error("Either --input-path (simple mode) or --template-a-path/--template-b-path with --p-train (bias mode) must be provided")
    
    # Process dataset
    train_path, val_path, test_path = process_dataset(
        input_path=args.input_path,
        template_a_path=args.template_a_path,
        template_b_path=args.template_b_path,
        text_column=args.text_column,
        label_column=args.label_column,
        template_column=args.template_column,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        p_train=args.p_train,
        target_size=args.target_size,
        max_samples_per_template=args.max_samples_per_template,
        group_column=args.group_column,
        group_map_path=args.group_map_path,
        group_map_key=args.group_map_key,
        stratify_groups=args.stratify_groups,
        match_rates=args.match_rates,
        p_eval=args.p_eval
    )
    
    print(f"\nData processing completed successfully!")
    print(f"Use these paths for training:")
    print(f"  --train-data {train_path}")
    print(f"  --val-data {val_path}")
    print(f"  --test-data {test_path}")

