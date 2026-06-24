import re
from collections import Counter
import argparse
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import load_phrasebank, prepare_splits_from_df
from .evaluate import compute_metrics, save_confusion_matrix
from .config import LABEL_MAP

PAD_ID, UNK_ID = 0, 1

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(train_texts, max_vocab_size=10000, min_freq=1):
    counter = Counter()
    for t in train_texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for word, freq in counter.most_common(max_vocab_size):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab, max_len=64):
    ids = [vocab.get(tok, UNK_ID) for tok in tokenize(text)][:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))



class PhraseBankDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=64):
        self.encoded = [encode(t, vocab, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long), self.labels[idx]
    


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3, pad_id=PAD_ID):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        final = torch.cat([h_n[0], h_n[1]], dim=1)  # concat final fwd + bwd hidden states
        return self.classifier(final)


def train_lstm_baseline(run_dir: Path, embed_dim=128, hidden_dim=128, lr=1e-3,
                          batch_size=32, num_epochs=10, max_len=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_phrasebank()
    train_df, val_df, test_df = prepare_splits_from_df(df)

    vocab = build_vocab(train_df["text"].tolist())  # train split ONLY

    def make_loader(split_df, shuffle):
        ds = PhraseBankDataset(split_df["text"].tolist(), split_df["label"].tolist(), vocab, max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_df, True)
    val_loader = make_loader(val_df, False)
    test_loader = make_loader(test_df, False)

    model = LSTMClassifier(len(vocab), embed_dim, hidden_dim, num_classes=len(LABEL_MAP)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    def run_eval(loader):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                logits = model(x.to(device))
                preds.append(logits.argmax(dim=-1).cpu().numpy())
                labels.extend(y.numpy())
        return compute_metrics(np.concatenate(preds), labels)

    best_val_f1, best_state = -1, None
    for epoch in range(1, num_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_metrics = run_eval(val_loader)
        print(f"epoch {epoch}: val_macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = run_eval(test_loader)

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "vocab.json").write_text(json.dumps(vocab))
    (run_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))

    test_preds, test_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            test_preds.append(model(x.to(device)).argmax(dim=-1).cpu().numpy())
            test_labels.extend(y.numpy())
    save_confusion_matrix(test_labels, np.concatenate(test_preds), run_dir / "confusion_matrix.png")

    print(json.dumps(test_metrics, indent=2))
    return test_metrics

def main():
    parser = argparse.ArgumentParser(description="Train a from-scratch BiLSTM baseline on Financial PhraseBank")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=64)
    args = parser.parse_args()
    train_lstm_baseline(Path(args.run_dir), args.embed_dim, args.hidden_dim,
                          args.learning_rate, args.batch_size, args.num_epochs, args.max_len)

if __name__ == "__main__":
    main()