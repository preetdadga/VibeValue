# 📈 VibeValue

**VibeValue** is a financial sentiment analysis project built using
[FinBERT](https://huggingface.co/yiyanghkust/finbert-pretrain).\
It classifies financial news and statements into three categories:
**positive, negative, neutral**.

------------------------------------------------------------------------

## 📂 Project Structure

    VibeValue/
    ├── notebooks/             # Jupyter/Colab notebooks
    │   └── vibevalue.ipynb
    ├── src/                   # Python scripts
    │   └── train.py
    ├── data/                  # Dataset (Financial PhraseBank)
    │   └── all-data.csv
    ├── requirements.txt       # Dependencies
    └── README.md

------------------------------------------------------------------------

## 🚀 Getting Started

### 1. Clone the Repository

``` bash
git clone https://github.com/<your-username>/VibeValue.git
cd VibeValue
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed, then run:

``` bash
pip install -r requirements.txt
```

### 3. Run Training & Evaluation

``` bash
python src/train.py
```

This will:\
- Load the dataset (`data/all-data.csv`)\
- Fine-tune **FinBERT**\
- Evaluate on validation & test sets\
- Save the trained model in `saved_model/`\
- Show a **confusion matrix** for predictions

------------------------------------------------------------------------

## 📊 Dataset

This project uses the **Financial PhraseBank** dataset.\
It consists of financial news sentences annotated for sentiment
(positive, negative, neutral).

-   **Source:** [Financial PhraseBank on Hugging
    Face](https://huggingface.co/datasets/financial_phrasebank)\
-   **Size:** \< 3 MB\
-   **Included:** already available in `data/all-data.csv`

------------------------------------------------------------------------

## 📑 Example Usage

To use the trained model for inference in Python:

``` python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained("saved_model")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")

# Example sentence
text = "The company's quarterly earnings exceeded expectations."

# Tokenize
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Predict
with torch.no_grad():
    logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=1).item()

label_map = {0: "positive", 1: "negative", 2: "neutral"}
print("Prediction:", label_map[prediction])
```

------------------------------------------------------------------------

## 📈 Results

During training, metrics such as **accuracy, precision, recall, and F1**
are reported.\
At the end, a **confusion matrix** is displayed to visualize model
performance across sentiment classes.

------------------------------------------------------------------------

## 🔮 Future Work

-   Hyperparameter tuning\
-   Experiment with other transformer models (RoBERTa, DistilBERT)\
-   Expand dataset with real-time financial news feeds
