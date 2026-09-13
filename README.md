# SentientAI — Sentiment Analysis Dashboard

A full-stack sentiment analysis system using LinearSVC + TF-IDF,
served via Flask with an Obsidian & Lime glassmorphism dashboard.

## Setup

```bash
cd sentiment_app
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open → http://localhost:5000

## Workflow

1. **Train tab** — point to your CSV path, set sample size, hit Start.
   - Default: `training.1600000.processed.noemoticon.csv` (Sentiment140)
   - 200k rows trains in ~2–3 minutes on a modern laptop
   - Model is saved to `model.pkl` and auto-loaded on restart

2. **Analyze tab** — type or paste any text for instant prediction.

3. **Batch tab** — paste multiple texts (one per line) for bulk analysis.

4. **Dashboard** — view accuracy, recent predictions, model metadata.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/status` | Training status + metrics |
| POST | `/api/train` | Start training `{csv_path, sample}` |
| POST | `/api/predict` | Predict `{texts: ["..."]}` |
| POST | `/api/batch` | Bulk predict `{content: "line1\nline2"}` |
| GET | `/api/history` | Last 50 predictions |
| POST | `/api/history/clear` | Clear history |

## Expected accuracy

| Model | Accuracy |
|-------|----------|
| Naive Bayes | ~76% |
| Logistic Regression | ~78% |
| **LinearSVC (this)** | **~81–83%** |
