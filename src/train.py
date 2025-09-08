import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# =========================
# 1. Load Dataset
# =========================
CSV_PATH = "data/all-data.csv"

df = pd.read_csv(CSV_PATH, encoding="latin1")
df.columns = ["label", "text"]

# Encode labels
label_map = {"positive": 0, "negative": 1, "neutral": 2}
df["label"] = df["label"].replace(label_map).astype(int)
df["text"] = df["text"].astype(str)

# Train/val/test split
df_train, df_test = train_test_split(df, stratify=df["label"], test_size=0.1, random_state=42)
df_train, df_val = train_test_split(df_train, stratify=df_train["label"], test_size=0.1, random_state=42)

print(f"Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")

# Convert to HuggingFace Datasets
train_ds = Dataset.from_pandas(df_train[["label", "text"]])
val_ds = Dataset.from_pandas(df_val[["label", "text"]])
test_ds = Dataset.from_pandas(df_test[["label", "text"]])

# =========================
# 2. Tokenization
# =========================
MODEL_NAME = "yiyanghkust/finbert-pretrain"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tok_func(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tok_func, batched=True).remove_columns("text")
val_ds = val_ds.map(tok_func, batched=True).remove_columns("text")
test_ds = test_ds.map(tok_func, batched=True).remove_columns("text")

train_ds.set_format(type="torch")
val_ds.set_format(type="torch")
test_ds.set_format(type="torch")

# =========================
# 3. Model & Training
# =========================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

batch_size = 32
lr = 2e-5
epochs = 5  # keep smaller for quick runs

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=torch.cuda.is_available(),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,
    weight_decay=0.01,
    report_to="none",
    num_train_epochs=epochs,
    load_best_model_at_end=True,
    logging_strategy="epoch"
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

# =========================
# 4. Confusion Matrix on Test Set
# =========================
predictions = trainer.predict(test_ds)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["positive", "negative", "neutral"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# =========================
# 5. Save Model
# =========================
trainer.save_model("saved_model")
print("✅ Model saved to saved_model/")
