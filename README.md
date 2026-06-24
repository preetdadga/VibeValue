# VibeValue

A financial sentiment classifier built by fine-tuning FinBERT on the Financial PhraseBank dataset, with an explicit experimental ladder isolating *why* it performs the way it does — not just a single accuracy number.

## Results

All in-domain numbers are macro-averaged over the same held-out test split (n=484; support 61 negative / 287 neutral / 136 positive), unless noted otherwise.

| Run | Base model | What it tests | Test macro F1 |
|---|---|---|---|
| `v6_lstm_baseline` | BiLSTM, randomly-initialized embeddings | Value of pretraining at all | 0.656 |
| `v1_baseline` | `yiyanghkust/finbert-pretrain` | Vanilla fine-tune, no imbalance handling | 0.809 |
| `v2_class_weighted` | `yiyanghkust/finbert-pretrain` | + class-weighted loss | **0.825** (best) |
| `v3_tuned` | `yiyanghkust/finbert-pretrain` | + more epochs, early stopping | 0.808 |
| `v5_prosus_benchmark` | `ProsusAI/finbert` (pretrained, not fine-tuned by us) | Reference comparison | 0.900* |

*\*Likely inflated — see [Benchmark contamination](#benchmark-contamination-v5) below. Not a fair number to compare against v1–v3 directly.*

**Cross-domain generalization**, evaluated on FiQA (n=1,173; combined train+valid+test splits), a dataset neither model below has been trained on:

| Run | Model evaluated | macro F1 |
|---|---|---|
| `v4_fiqa_yours` | `v2_class_weighted` | 0.416 |
| `v4_fiqa_prosus` | `ProsusAI/finbert` | 0.480 |

### What these numbers actually show

1. **Domain pretraining matters, a lot.** Swapping a from-scratch BiLSTM for a financial-domain-pretrained transformer — same data, same split, same classification head — buys +0.153 macro F1 (v6 → v1). This isolates the value of pretraining specifically, since everything else is held constant.
2. **Class weighting helps, modestly and honestly.** v2 beats v1 by +0.016 macro F1. The improvement has the expected shape: negative-class recall rises (0.721 → 0.803) at a small cost to neutral recall and positive precision — the textbook reweighting trade-off, just smaller than we initially (incorrectly) measured. An earlier pass at this comparison showed a much larger gap, which turned out to be a confound: that v1 run had accidentally trained on only a 500-row debug subset instead of the full dataset. Re-running v1 on the full data with identical hyperparameters corrected the comparison to the modest, trustworthy +0.016 above.
3. **More epochs didn't help.** v3 (6 epochs, early stopping enabled) scored essentially the same as v2 (2 epochs) on the test set, despite validation macro F1 peaking slightly higher at epoch 4 (0.8415) during that run. The gap between that validation peak and the eventual test score is ordinary sampling noise at this dataset size (test set is 484 examples), not a methodology problem — confirmed by inspecting the `EarlyStoppingCallback`/`load_best_model_at_end` configuration directly rather than assuming it was misconfigured.
4. **Both models share the same out-of-domain weakness, and it's explainable.** On FiQA, both `v2_class_weighted` and `ProsusAI/finbert` show the same failure pattern: very poor neutral precision (~10%) despite decent neutral recall, and high positive precision but badly under-called positive recall. Financial PhraseBank is 59% neutral; FiQA is only 8% neutral and 61% positive — nearly inverted. Both models appear to have inherited PhraseBank's class-distribution prior, and that prior actively misleads them on FiQA's very different distribution. Since the *same* pattern appears in a model neither we nor Prosus could have influenced identically, this points to a structural domain-transfer limitation, not something specific to our training choices.

### Benchmark contamination (v5)

`ProsusAI/finbert`'s own model card states it was fine-tuned using Financial PhraseBank (Malo et al., 2014) — the same dataset our test split is drawn from. That means its strong score above (0.900) almost certainly benefits from having already seen some of the sentences in *our* test split during its own original training. The FiQA cross-domain comparison above is the fairer benchmark against this model, since FiQA is independent of Financial PhraseBank entirely.

This is also why `v1`–`v3` use `yiyanghkust/finbert-pretrain` rather than `ProsusAI/finbert` as the fine-tuning base: `yiyanghkust/finbert-pretrain` is financial-domain pretrained but has no prior exposure to Financial PhraseBank's sentiment labels, so fine-tuning on it doesn't carry the same contamination risk.

## Repo structure

```
vibevalue/
├── pyproject.toml
├── data/raw/all-data.csv       # Financial PhraseBank, latin-1, no header
├── src/vibevalue/
│   ├── config.py                # LABEL_MAP, dataset path, seed — single source of truth
│   ├── data.py                  # load, dedupe, validate, split
│   ├── train.py                 # fine-tuning entrypoint (v1/v2/v3)
│   ├── evaluate.py               # standalone eval on a saved or Hub model
│   ├── fiqa_eval.py               # cross-domain eval against FiQA (v4)
│   ├── lstm_baseline.py            # from-scratch BiLSTM baseline (v6)
│   └── serve.py                     # FastAPI serving app
├── runs/
│   ├── debug/                    # smoke-test runs only — kept separate from real versions
│   └── v1_baseline/, v2_class_weighted/, ...  # one directory per real experiment
├── tests/
```

Each `runs/<name>/` directory contains the saved model, `metrics.json`, `confusion_matrix.png`, and run notes.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements/base.txt -r requirements/train.txt -r requirements/serve.txt
```

**GPU note:** a plain `pip install torch` may resolve to a CPU-only wheel depending on your platform/pip configuration. Verify before training:

```powershell
.venv\Scripts\python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

If `cuda` prints `None`, reinstall explicitly:

```powershell
.venv\Scripts\python -m pip uninstall torch -y
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

(check [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the current correct index tag for your CUDA version — these change over time.)

**Data:** place the raw Financial PhraseBank CSV at `data/raw/all-data.csv` (latin-1 encoded, no header, two columns `label,text`). The loader handles this format directly; no preprocessing needed beforehand. 4,846 raw rows, 6 exact duplicates dropped automatically at load time → 4,840 used for training (negative: 604, neutral: 2,873, positive: 1,363).

**Running commands:** every command below assumes `PYTHONPATH` includes `src/`:

```powershell
$env:PYTHONPATH='src'
```

## Usage

**Train a model:**
```powershell
python -m vibevalue.train --run-dir runs/my_run --base-model yiyanghkust/finbert-pretrain --learning-rate 2e-5 --batch-size 16 --num-epochs 2 [--class-weighted] --notes "describe what changed"
```

**Evaluate a saved (or Hub) model on the Financial PhraseBank test split:**
```powershell
python -m vibevalue.evaluate --model-dir runs/my_run/model --output-dir runs/my_run
```
Works with a Hub repo id too (e.g. `--model-dir ProsusAI/finbert`) — handles label-id remapping automatically if the model's own `id2label` ordering differs from this project's (`ProsusAI/finbert` uses `0=positive,1=negative,2=neutral`; this project uses `0=negative,1=neutral,2=positive`).

**Evaluate cross-domain on FiQA:**
```powershell
python -m vibevalue.fiqa_eval --model-dir runs/my_run/model --output-dir runs/my_run_fiqa
```
Loads `TheFinAI/fiqa-sentiment-classification` from the Hub, converts its continuous −1..+1 sentiment score to this project's 3-class labels using thresholds with precedent in prior financial-NLP work (`score < -0.1` → negative, `-0.1` to `0.1` → neutral, `score >= 0.1` → positive), and evaluates against the combined train+valid+test rows (1,173 total) as one held-out set.

**Train the from-scratch LSTM baseline:**
```powershell
python -m vibevalue.lstm_baseline --run-dir runs/v6_lstm_baseline
```

## Dataset & citations

- **Financial PhraseBank**: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., Takala, P. (2014). "Good debt or bad debt: Detecting semantic orientations in economic texts." Licensed **CC-BY-NC-SA-3.0 — non-commercial use only**.
- **FiQA Task 1**: Maia, M. et al. (2018). "WWW'18 Open Challenge: Financial Opinion Mining and Question Answering" / "A Baseline for Aspect-Based Sentiment Analysis in Financial Microblogs and News." Used here purely as a held-out cross-domain eval set, never for training.
- **`yiyanghkust/finbert-pretrain`**: Yang, Y., Uy, M.C.S., Huang, A. (2020). "FinBERT: A Pretrained Language Model for Financial Communications." arXiv:2006.08097.
- **`ProsusAI/finbert`**: community FinBERT checkpoint, fine-tuned on Financial PhraseBank by Prosus (per its own model card) — used here only as a frozen reference, never retrained.

## Engineering notes

A few non-obvious issues surfaced while building this, worth documenting since they're easy to hit again:

- **`transformers` API drift across versions**: `evaluation_strategy` was renamed `eval_strategy`; `Trainer(tokenizer=...)` was renamed to `processing_class=...`; `accelerate>=1.1.0` is required but not always pulled in automatically; custom `compute_loss` overrides need to accept `**kwargs` for forward-compatibility, since `Trainer.training_step` started passing additional arguments (e.g. `num_items_in_batch`) that older custom subclasses didn't expect.
- **`yiyanghkust/finbert-pretrain` is incompatible with `AutoTokenizer`/`AutoModelForSequenceClassification`** on current `transformers` — its repo files predate conventions the `Auto*` classes now expect (missing fast-tokenizer files, missing `model_type` in `config.json`). Fixed by loading with the concrete classes directly: `BertTokenizer.from_pretrained(...)` / `BertForSequenceClassification.from_pretrained(...)`.
- **Label-id ordering is not universal across checkpoints.** This project uses `0=negative,1=neutral,2=positive`; `ProsusAI/finbert` uses a different order. `evaluate.py` and `fiqa_eval.py` both check each loaded model's own `config.id2label` and remap predictions into this project's canonical ordering before scoring — silently skipping this step would have produced confidently-wrong accuracy numbers.
- **`pathlib.Path()` mangles Hugging Face Hub repo ids on Windows** — `Path("ProsusAI/finbert")` renders with a backslash when converted back to a string, which Hugging Face doesn't recognize as a valid repo id. Pass repo ids as plain strings, not through `Path()`.
- **Debug runs and real versioned runs should never share a directory name.** Reusing `runs/v1_baseline` for both a CPU smoke-test debug run and the real full-dataset run caused a confound that wasn't caught until configs were compared side-by-side. Debug runs are now routed to a separate `runs/debug/` namespace, kept entirely apart from the real versioned runs.

## Limitations & future work

- Out-of-domain generalization (FiQA) is a known, explained weakness — the model inherits Financial PhraseBank's class-distribution prior. A documented follow-up worth doing: a `v7` experiment that deliberately mixes a small amount of FiQA-style training data in, to test whether it corrects the prior without hurting in-domain performance.
- `Dockerfile.train` / `Dockerfile.serve` exist per the original design (separate train/serve dependency sets, model artifacts mounted as volumes rather than baked in) but weren't exercised end-to-end during this build session — verify the build and run before relying on them for a demo.
- Serving (`serve.py`) follows the same caveat — confirm the `/predict` endpoint actually works against a real saved model before treating it as done.
- No CI pipeline yet; `tests/` covers data loading/validation and metrics computation only, and runs locally via `pytest` with `PYTHONPATH=src` set.
