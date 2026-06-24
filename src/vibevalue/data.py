"""Data loading, validation, and splitting utilities for VibeValue."""
from pathlib import Path
from typing import Tuple
import pandas as pd

from .config import LABEL_MAP, LABEL_NAME_TO_ID, HF_DATASET, RAW_DATA_PATH, SEED


def load_from_csv(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load Financial PhraseBank from a fallback CSV file and map labels to ids."""
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV data file not found at {path}")

    df = pd.read_csv(path, encoding="latin1", header=None, names=["label", "text"])
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("Raw CSV file must contain two columns: label and text")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()

    invalid_labels = set(df["label"].unique()) - set(LABEL_NAME_TO_ID.keys())
    if invalid_labels:
        raise ValueError(f"Found invalid label values in CSV: {invalid_labels}")

    df["label"] = df["label"].map(LABEL_NAME_TO_ID)
    return df


def load_hf_dataset(config_name=HF_DATASET):
    """Load the Financial PhraseBank from Hugging Face and return a pandas DataFrame.

    This function lazy-imports `datasets` to avoid failing when network/tests don't need it.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("datasets library is required to load HF datasets") from e

    ds = load_dataset(*config_name)
    df = pd.DataFrame(ds["train"])
    if "sentence" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"sentence": "text"})
    return df


def load_phrasebank() -> pd.DataFrame:
    """Load the Financial PhraseBank and normalize labels and text."""
    if RAW_DATA_PATH.exists():
        df = load_from_csv(RAW_DATA_PATH)
    else:
        df = load_hf_dataset()

    if "sentence" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"sentence": "text"})

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    dup_count = df.duplicated(subset=["text", "label"]).sum()
    if dup_count:
        print(f"Found {dup_count} exact duplicate rows; dropping duplicates.")
        df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    return df


def validate_df(df: pd.DataFrame):
    """Validate DataFrame has correct label ids and non-empty text.

    Raises ValueError on problems.
    Returns duplicate count.
    """
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("DataFrame must contain 'text' and 'label' columns")

    labels = set(df["label"].unique().tolist())
    allowed = set(LABEL_MAP.keys())
    bad = labels - allowed
    if bad:
        raise ValueError(f"Found invalid label ids: {bad}")

    df["text"] = df["text"].astype(str).str.strip()
    empty = df["text"].eq("").sum()
    if empty:
        raise ValueError(f"Found {empty} empty text rows after stripping")

    dup_count = df.duplicated(subset=["text", "label"]).sum()
    return int(dup_count)


def prepare_splits_from_df(df: pd.DataFrame, seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate and produce stratified 80/10/10 train/val/test splits.

    Returns (train, val, test)
    """
    dup_count = validate_df(df)

    from sklearn.model_selection import train_test_split

    # ensure reproducible ordering
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # first split off 20% for val+test
    try:
        train, temp = train_test_split(
            df, test_size=0.2, stratify=df["label"], random_state=seed
        )

        # split temp into val and test equally (10% each of original)
        val, test = train_test_split(
            temp, test_size=0.5, stratify=temp["label"], random_state=seed
        )
    except ValueError:
        # Fall back to non-stratified split for tiny datasets where stratify fails
        train, temp = train_test_split(df, test_size=0.2, random_state=seed)
        val, test = train_test_split(temp, test_size=0.5, random_state=seed)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
