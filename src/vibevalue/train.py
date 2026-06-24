"""Training entrypoint for VibeValue."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import LABEL_MAP, SEED
from .data import load_phrasebank, prepare_splits_from_df
from .evaluate import compute_metrics


class PhrasebankDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=len(LABEL_MAP))
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights * (len(LABEL_MAP) / weights.sum())
    return torch.tensor(weights, dtype=torch.float)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def train(run_dir: Path, base_model: str, learning_rate: float, batch_size: int, num_epochs: int, class_weighted: bool, notes: str, debug_subset: int = None):
    print(f"Loading phrasebank data from fallback or HF...")
    df = load_phrasebank()
    print(f"Raw loaded rows: {len(df)}")
    train_df, val_df, test_df = prepare_splits_from_df(df)
    if debug_subset:
        train_df = train_df.sample(n=debug_subset, random_state=SEED).reset_index(drop=True)
        val_df = val_df.sample(n=min(debug_subset, len(val_df)), random_state=SEED).reset_index(drop=True)
        test_df = test_df.sample(n=min(debug_subset, len(test_df)), random_state=SEED).reset_index(drop=True)

    tokenizer = BertTokenizer.from_pretrained(base_model)
    train_dataset = PhrasebankDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_dataset = PhrasebankDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer)
    test_dataset = PhrasebankDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)

    model = BertForSequenceClassification.from_pretrained(base_model, num_labels=len(LABEL_MAP))
    model_dir = run_dir / "model"
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        logging_strategy="epoch",
        report_to=[],
    )

    set_seed(SEED)
    class_weights = compute_class_weights(train_df["label"].tolist()) if class_weighted else None
    if class_weighted:
        print(f"Computed class weights: {class_weights.tolist()}")

    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", {
        "base_model": base_model,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "class_weighted": class_weighted,
        "debug_subset": debug_subset,
        "seed": SEED,
    })

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: compute_metrics(np.argmax(p.predictions, axis=1), p.label_ids),
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("Evaluating on test split...")
    test_pred = trainer.predict(test_dataset)
    test_metrics = compute_metrics(np.argmax(test_pred.predictions, axis=1), test_pred.label_ids)
    save_json(run_dir / "metrics.json", test_metrics)

    from .evaluate import save_confusion_matrix
    save_confusion_matrix(test_pred.label_ids, np.argmax(test_pred.predictions, axis=1), run_dir / "confusion_matrix.png")

    note_text = notes or "Trained a FinBERT-based sentiment classifier on Financial PhraseBank."
    (run_dir / "NOTES.md").write_text(note_text, encoding="utf-8")
    print(f"Run completed. Artifacts saved to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train VibeValue sentiment classifier")
    parser.add_argument("--run-dir", default="runs/v1_baseline")
    parser.add_argument("--base-model", default="ProsusAI/finbert")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--class-weighted", action="store_true")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--debug-subset", type=int, default=0)
    args = parser.parse_args()
    train(
        run_dir=Path(args.run_dir),
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        class_weighted=args.class_weighted,
        notes=args.notes,
        debug_subset=args.debug_subset or None,
    )


if __name__ == "__main__":
    main()
