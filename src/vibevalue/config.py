"""Configuration and label map for VibeValue."""
from pathlib import Path

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

LABEL_NAME_TO_ID = {v: k for k, v in LABEL_MAP.items()}

HF_DATASET = ("takala/financial_phrasebank", "sentences_50agree")
RAW_DATA_PATH = Path("data/raw/all-data.csv")
SEED = 42
