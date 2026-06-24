"""FastAPI serving app for VibeValue."""
import os
from functools import lru_cache
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from .config import LABEL_MAP

app = FastAPI()


class PredictIn(BaseModel):
    text: str


@lru_cache(maxsize=1)
def get_model_and_tokenizer():
    model_dir = os.getenv("VIBEVALUE_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("VIBEVALUE_MODEL_DIR is not set")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictIn):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="text too long")

    tokenizer, model = get_model_and_tokenizer()
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0].cpu()
    probs = torch.softmax(logits, dim=-1).tolist()

    scores = {LABEL_MAP[idx]: float(probs[idx]) for idx in range(len(LABEL_MAP))}
    best = max(scores, key=scores.get)
    return {
        "label": best,
        "score": scores[best],
        "scores": scores,
    }
