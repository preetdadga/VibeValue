"""Evaluation utilities for VibeValue."""
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

from .config import LABEL_MAP, LABEL_NAME_TO_ID
from .data import load_phrasebank, prepare_splits_from_df


def compute_metrics(preds, labels) -> Dict[str, Any]:
    """Compute per-class precision/recall/f1 and overall accuracy.
    preds/labels are 1D arrays of integer class ids.
    """
    preds = np.array(preds)
    labels = np.array(labels)

    accuracy = float(accuracy_score(labels, preds))
    p, r, f1, support = precision_recall_fscore_support(labels, preds, labels=list(LABEL_MAP.keys()), zero_division=0)

    results = {"accuracy": accuracy}
    for idx, cls in enumerate(LABEL_MAP.keys()):
        results[f"precision_{cls}"] = float(p[idx])
        results[f"recall_{cls}"] = float(r[idx])
        results[f"f1_{cls}"] = float(f1[idx])
        results[f"support_{cls}"] = int(support[idx])

    results["macro_f1"] = float(np.mean(f1))
    return results


def save_confusion_matrix(true_labels, pred_labels, path: Path):
    labels = list(LABEL_MAP.keys())
    matrix = confusion_matrix(true_labels, pred_labels, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=[LABEL_MAP[x] for x in labels],
        yticklabels=[LABEL_MAP[x] for x in labels],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def load_model_and_tokenizer(model_dir: Path):
    from transformers import AutoModelForSequenceClassification, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model


def evaluate_model(model_dir: Path, output_dir: Path):
    import torch

    tokenizer, model = load_model_and_tokenizer(model_dir)
    df = load_phrasebank()
    _, _, test_df = prepare_splits_from_df(df)

    inputs = tokenizer(
        test_df["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.argmax(dim=-1).cpu().numpy()
    # Some checkpoints (e.g. a different pretrained model) use a different
    # id<->label ordering than this project's LABEL_MAP. Remap if so.
    model_id2label = {int(k): v for k, v in model.config.id2label.items()}
    if set(model_id2label.values()) <= set(LABEL_NAME_TO_ID.keys()):
        remap = {model_id: LABEL_NAME_TO_ID[name] for model_id, name in model_id2label.items()}
        preds = np.array([remap[p] for p in preds])
    # else: assume the model's ids already match LABEL_MAP (true for anything
    # trained directly through this project's train.py)
    metrics = compute_metrics(preds, test_df["label"].tolist())
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_matrix(test_df["label"].tolist(), preds, output_dir / "confusion_matrix.png")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved VibeValue model")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_dir).parent
    metrics = evaluate_model(args.model_dir, output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
