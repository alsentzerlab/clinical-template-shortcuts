from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class DataColumns:
    text: str = "text"
    readmission: str = "label"
    template: str = "template"


def load_dataset_from_splits(
    *,
    train_path: str,
    validation_path: str,
    test_path: str,
    columns: DataColumns = DataColumns(),
) -> DatasetDict:
    """Load the exact train/validation/test Parquet splits we already have."""

    def load_one(path: str) -> Dataset:
        df = pd.read_parquet(path)
        df = df.dropna(subset=[columns.text, columns.readmission, columns.template])
        unexpected = set(df[columns.template].unique()) - {"A", "B"}
        if unexpected:
            # Otherwise these rows would silently map to NaN and be dropped.
            raise ValueError(f"{path}: template column must be 'A'/'B', found {sorted(map(str, unexpected))[:5]}")
        df[columns.template] = df[columns.template].map({"A": 0, "B": 1})

        if df[columns.readmission].dtype != "int64":
            df[columns.readmission] = df[columns.readmission].astype(int)
        if df[columns.template].dtype != "int64":
            df[columns.template] = df[columns.template].astype(int)

        df = df[[columns.text, columns.readmission, columns.template]].reset_index(drop=True)
        ds = Dataset.from_pandas(df, preserve_index=False)

        def rename_labels(ex):
            return {
                "text": ex[columns.text],
                # Order: [readmission, template]
                "labels": [ex[columns.readmission], ex[columns.template]],
            }

        return ds.map(rename_labels, remove_columns=ds.column_names)

    return DatasetDict(
        train=load_one(train_path),
        validation=load_one(validation_path),
        test=load_one(test_path),
    )


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: Optional[int] = 512,
    truncation: bool = True,
) -> DatasetDict:
    """max_length=None reads full notes (no truncation), as in finetuning."""

    def tok(batch):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=truncation and max_length is not None,
        )

    tokenized = dataset.map(tok, batched=True, remove_columns=["text"])
    return tokenized

