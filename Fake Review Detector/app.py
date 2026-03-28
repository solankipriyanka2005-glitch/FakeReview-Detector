# app.py
import os
import re
import string
import pickle
import numpy as np
from collections import Counter
from flask import Flask, render_template, request
from lime.lime_text import LimeTextExplainer

APP_TITLE = "🕵️ Fake Review Detector"

MODEL_DIR = "models"

VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
MODEL_FILES = {
    "Naive Bayes":         os.path.join(MODEL_DIR, "naive_bayes_model.pkl"),
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression_model.pkl"),
    "Random Forest":       os.path.join(MODEL_DIR, "random_forest_model.pkl"),
    "SVM":                 os.path.join(MODEL_DIR, "svm_model.pkl"),
}

# --------------------
# Preprocess (same as training)
# --------------------
def clean_text(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\s+", " ", t).strip()
    return t

# --------------------
# Load artifacts
# --------------------
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

models = {}
for name, path in MODEL_FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")
    with open(path, "rb") as f:
        models[name] = pickle.load(f)

# For LIME, we’ll use Logistic Regression (well-calibrated, linear)
lr_model = models["Logistic Regression"]
lr_classes = list(lr_model.classes_)  # e.g. ["Fake","Real"] in some order

# Ensure both 'Fake' and 'Real' exist
if not all(lbl in lr_classes for lbl in ["Fake", "Real"]):
    raise ValueError(f"Logistic Regression classes must contain 'Fake' and 'Real'. Got {lr_classes}")

# LIME expects predict_proba that takes raw texts.
def lr_predict_proba_for_lime(raw_texts):
    cleaned = [clean_text(t) for t in raw_texts]
    X = vectorizer.transform(cleaned)
    proba = lr_model.predict_proba(X)  # columns follow lr_model.classes_
    # Reorder to [Fake, Real] for explainer (below)
    idx_fake = lr_classes.index("Fake")
    idx_real = lr_classes.index("Real")
    return np.column_stack([proba[:, idx_fake], proba[:, idx_real]])

explainer = LimeTextExplainer(class_names=["Fake", "Real"])

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    review = ""
    results = {}        # per-model: {prediction, confidence}
    prob_summary = {}   # chart values (confidence % for predicted class)
    vote_counts = {"Real": 0, "Fake": 0}
    final_decision = None
    final_confidence = None
    lime_html = None

    if request.method == "POST":
        review = (request.form.get("review") or "").strip()
        cleaned = clean_text(review)
        X = vectorizer.transform([cleaned])

        votes = []
        confs = []

        for name, model in models.items():
            pred = model.predict(X)[0]                 # "Real" or "Fake" (string)
            probs = model.predict_proba(X)[0]          # aligned to model.classes_
            classes = list(model.classes_)
            p_idx = classes.index(pred)
            p_conf = float(probs[p_idx])               # 0..1

            results[name] = {
                "prediction": pred,
                "confidence": round(p_conf * 100, 2)
            }
            prob_summary[name] = round(p_conf * 100, 2)
            votes.append(pred)
            confs.append(p_conf)

        # Majority vote
        c = Counter(votes)
        vote_counts = {"Real": c.get("Real", 0), "Fake": c.get("Fake", 0)}
        if vote_counts["Real"] > vote_counts["Fake"]:
            final_decision = "Real"
        elif vote_counts["Fake"] > vote_counts["Real"]:
            final_decision = "Fake"
        else:
            # tie -> higher average confidence wins
            avg_r = np.mean([conf for v, conf in zip(votes, confs) if v == "Real"]) if vote_counts["Real"] else 0.0
            avg_f = np.mean([conf for v, conf in zip(votes, confs) if v == "Fake"]) if vote_counts["Fake"] else 0.0
            final_decision = "Real" if avg_r >= avg_f else "Fake"

        # Average confidence of agreeing models
        agreeing = [conf for v, conf in zip(votes, confs) if v == final_decision]
        final_confidence = round(float(np.mean(agreeing) * 100), 2) if agreeing else 0.0

        # LIME explanation (LogReg)
        lime_exp = explainer.explain_instance(
            review,
            lr_predict_proba_for_lime,
            num_features=10
        )
        lime_html = lime_exp.as_html()

    return render_template(
        "index.html",
        app_title=APP_TITLE,
        review=review,
        results=results,
        prob_summary=prob_summary,
        vote_counts=vote_counts,
        final_decision=final_decision,
        final_confidence=final_confidence,
        lime_html=lime_html
    )

if __name__ == "__main__":
    app.run(debug=True)
