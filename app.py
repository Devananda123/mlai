import os
import re
import time
import pickle
import threading
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

app = Flask(__name__)
CORS(app)

# Global state
model = None
vectorizer = None
training_status = {
    "state": "idle",       # idle | training | ready | error
    "message": "",
    "progress": 0,
    "accuracy": None,
    "report": None,
    "trained_at": None,
    "sample_count": 0,
}
history = []   # list of {text, prediction, confidence, timestamp}

# ── helpers ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()


def train_model_thread(csv_path: str, sample: int):
    global model, vectorizer, training_status

    try:
        training_status.update(state="training", message="Loading dataset…", progress=5)

        df = pd.read_csv(csv_path, encoding='latin-1', header=None)
        df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
        df['target'] = df['target'].map({0: 0, 4: 1})
        df = df.dropna(subset=['text', 'target'])

        if sample and sample < len(df):
            df = df.sample(n=sample, random_state=42)

        training_status.update(message="Cleaning text…", progress=20, sample_count=len(df))

        df['clean'] = df['text'].apply(clean_text)

        training_status.update(message="Vectorizing with TF-IDF…", progress=40)

        X_train, X_test, y_train, y_test = train_test_split(
            df['clean'], df['target'], test_size=0.2, random_state=42)

        vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        X_tr = vec.fit_transform(X_train)
        X_te = vec.transform(X_test)

        training_status.update(message="Training LinearSVC…", progress=60)

        svm = LinearSVC(C=1.0, max_iter=2000)
        svm.fit(X_tr, y_train)

        training_status.update(message="Evaluating…", progress=85)

        preds = svm.predict(X_te)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds).tolist()

        model = svm
        vectorizer = vec

        # Persist
        with open('model.pkl', 'wb') as f:
            pickle.dump(svm, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vec, f)

        training_status.update(
            state="ready",
            message="Model ready",
            progress=100,
            accuracy=round(acc * 100, 2),
            report=report,
            confusion_matrix=cm,
            trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    except Exception as e:
        training_status.update(state="error", message=str(e), progress=0)


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/status')
def status():
    return jsonify(training_status)


@app.route('/api/train', methods=['POST'])
def train():
    if training_status['state'] == 'training':
        return jsonify({"error": "Training already in progress"}), 400

    data = request.json or {}
    csv_path = data.get('csv_path', 'training.1600000.processed.noemoticon.csv')
    sample = int(data.get('sample', 200000))

    if not os.path.exists(csv_path):
        return jsonify({"error": f"CSV not found: {csv_path}"}), 400

    t = threading.Thread(target=train_model_thread, args=(csv_path, sample), daemon=True)
    t.start()
    return jsonify({"message": "Training started"})


@app.route('/api/predict', methods=['POST'])
def predict():
    global history, model, vectorizer

    if model is None or vectorizer is None:
        # Try loading persisted model
        if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            training_status['state'] = 'ready'
        else:
            return jsonify({"error": "Model not trained yet"}), 400

    data = request.json or {}
    texts = data.get('texts', [])
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return jsonify({"error": "No text provided"}), 400

    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)

    # Decision function score → pseudo-confidence via sigmoid
    scores = model.decision_function(X)
    confidences = (1 / (1 + np.exp(-np.abs(scores)))).tolist()

    results = []
    for i, text in enumerate(texts):
        label = "positive" if preds[i] == 1 else "negative"
        conf = round(confidences[i] * 100, 1)
        entry = {
            "text": text,
            "prediction": label,
            "confidence": conf,
            "timestamp": time.strftime("%H:%M:%S"),
        }
        results.append(entry)
        history.insert(0, entry)

    history = history[:50]  # keep last 50
    return jsonify({"results": results})


@app.route('/api/history')
def get_history():
    return jsonify(history)


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    global history
    history = []
    return jsonify({"message": "History cleared"})


@app.route('/api/batch', methods=['POST'])
def batch():
    """Batch predict from uploaded text (newline-separated)."""
    if model is None:
        return jsonify({"error": "Model not trained yet"}), 400

    data = request.json or {}
    raw = data.get('content', '')
    lines = [l.strip() for l in raw.split('\n') if l.strip()]

    if not lines:
        return jsonify({"error": "No lines provided"}), 400

    cleaned = [clean_text(l) for l in lines]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)
    scores = model.decision_function(X)
    confidences = (1 / (1 + np.exp(-np.abs(scores)))).tolist()

    pos = sum(1 for p in preds if p == 1)
    neg = len(preds) - pos

    results = [
        {
            "text": lines[i],
            "prediction": "positive" if preds[i] == 1 else "negative",
            "confidence": round(confidences[i] * 100, 1),
        }
        for i in range(len(lines))
    ]

    return jsonify({
        "total": len(lines),
        "positive": pos,
        "negative": neg,
        "results": results,
    })


if __name__ == '__main__':
    # Auto-load persisted model on startup
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        training_status['state'] = 'ready'
        training_status['message'] = 'Model loaded from disk'
        print("✓ Loaded persisted model")

    app.run(debug=True, port=5000)