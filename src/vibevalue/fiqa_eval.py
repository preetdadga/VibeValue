from .config import LABEL_NAME_TO_ID
import argparse, json
import numpy as np
from pathlib import Path

from .config import LABEL_MAP
from .evaluate import compute_metrics, save_confusion_matrix

def score_to_label_id(score: float) -> int:
    if score < -0.1:
        name = "negative"
    elif score >= 0.1:
        name = "positive"
    else:
        name = "neutral"
    return LABEL_NAME_TO_ID[name]

def load_fiqa():
    from datasets import load_dataset, concatenate_datasets

    ds = load_dataset("TheFinAI/fiqa-sentiment-classification")
    combined = concatenate_datasets([ds["train"], ds["valid"], ds["test"]])
    sentences = combined["sentence"]
    labels = [score_to_label_id(s) for s in combined["score"]]
    return sentences, labels


def evaluate_on_fiqa(model_dir: str, output_dir: Path, batch_size: int = 32):
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    sentences, labels = load_fiqa()

    all_preds = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        all_preds.append(outputs.logits.argmax(dim=-1).cpu().numpy())
    preds = np.concatenate(all_preds)

    model_id2label = {int(k): v for k, v in model.config.id2label.items()}
    if set(model_id2label.values()) <= set(LABEL_NAME_TO_ID.keys()):
        remap = {mid: LABEL_NAME_TO_ID[name] for mid, name in model_id2label.items()}
        preds = np.array([remap[p] for p in preds])

    metrics = compute_metrics(preds, labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_matrix(labels, preds, output_dir / "confusion_matrix.png")
    return metrics



def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the FiQA cross-domain benchmark")
    parser.add_argument("--model-dir", required=True, help="Local path or HF Hub repo id")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metrics = evaluate_on_fiqa(args.model_dir, Path(args.output_dir))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()