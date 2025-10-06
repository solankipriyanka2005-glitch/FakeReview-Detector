# train.py
import os
import re
import string
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# -------------------------
# CONFIG — change if needed
# -------------------------
CSV_PATH = "deceptive-opinion.csv"      # your uploaded file
TEXT_COL_CANDIDATES  = ["text", "review", "review_text", "content", "sentence"]
LABEL_COL_CANDIDATES = ["label", "gold", "truth", "ground_truth"]

SAVE_DIR = "models"
VECTORIZER_PATH = os.path.join(SAVE_DIR, "vectorizer.pkl")
MODEL_PATHS = {
    "naive_bayes":          os.path.join(SAVE_DIR, "naive_bayes_model.pkl"),
    "logistic_regression":  os.path.join(SAVE_DIR, "logistic_regression_model.pkl"),
    "random_forest":        os.path.join(SAVE_DIR, "random_forest_model.pkl"),
    "svm":                  os.path.join(SAVE_DIR, "svm_model.pkl"),
}

# ---------------------------------
# 1) Load CSV and detect columns
# ---------------------------------
df = pd.read_csv(CSV_PATH)

def pick_col(candidates, cols):
    for c in candidates:
        if c in cols:
            return c
    return None

text_col  = pick_col(TEXT_COL_CANDIDATES,  [c.lower() for c in df.columns])
label_col = pick_col(LABEL_COL_CANDIDATES, [c.lower() for c in df.columns])

# Map to actual column names
col_map = {c.lower(): c for c in df.columns}
if text_col is None or label_col is None:
    raise ValueError(
        f"Could not find text/label columns. Found columns: {list(df.columns)}\n"
        f"Expected one of text in {TEXT_COL_CANDIDATES} and label in {LABEL_COL_CANDIDATES}."
    )

TEXT_COL  = col_map[text_col]
LABEL_COL = col_map[label_col]
df = df[[TEXT_COL, LABEL_COL]].dropna()

# ------------------------------------------------
# 2) Normalize labels -> 'Real' and 'Fake' (strings)
# ------------------------------------------------
def normalize_label(v):
    s = str(v).strip().lower()
    if s in {"deceptive", "fake", "spam", "promotional", "fraud", "not truthful", "false"}:
        return "Fake"
    if s in {"truthful", "genuine", "real", "non-spam", "legit", "original"}:
        return "Real"
    # numeric common cases
    if s == "0": return "Fake"
    if s == "1": return "Real"
    # if it's already exactly "real" or "fake" capitalization variants
    if s == "fake": return "Fake"
    if s == "real": return "Real"
    raise ValueError(f"Unrecognized label value: {v}")

df["label_norm"] = df[LABEL_COL].apply(normalize_label)

# ---------------------------------
# 3) Clean text (simple + consistent)
# ---------------------------------
def clean_text(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"<[^>]+>", " ", t)            # remove HTML tags
    t = re.sub(r"\d+", " ", t)                # remove numbers
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["text_clean"] = df[TEXT_COL].astype(str).apply(clean_text)

# ---------------------------------
# 4) Split & Vectorize
# ---------------------------------
X = df["text_clean"].values
y = df["label_norm"].values  # 'Real' or 'Fake'

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    max_features=20000
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ---------------------------------
# 5) Train models (all with proba)
# ---------------------------------
models = {}

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
models["naive_bayes"] = nb

# Logistic Regression (balanced, more iterations)
lr = LogisticRegression(max_iter=2000, class_weight="balanced")
lr.fit(X_train_vec, y_train)
models["logistic_regression"] = lr

# Random Forest (more trees, balanced)
rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1
)
rf.fit(X_train_vec, y_train)
models["random_forest"] = rf

# SVM (LinearSVC + probability via calibration)
lsvc = LinearSVC(dual=True, max_iter=5000)
svm = CalibratedClassifierCV(lsvc, method="sigmoid", cv=5)
svm.fit(X_train_vec, y_train)
models["svm"] = svm

# ---------------------------------
# 6) Evaluate
# ---------------------------------
print("\n================= EVALUATION =================")
for name, mdl in models.items():
    preds = mdl.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name}: Accuracy = {acc:.4f}")
    print(classification_report(y_test, preds, digits=4))

# Optional macro AUC if classes > 1 and proba exists
try:
    # align to 'Fake','Real' order for the sake of AUC (binary)
    # encode labels -> Fake=0, Real=1
    enc = LabelEncoder()
    enc.fit(["Fake", "Real"])
    y_true = enc.transform(y_test)
    y_proba = models["logistic_regression"].predict_proba(X_test_vec)
    # columns follow classes_ order
    cls_order = list(models["logistic_regression"].classes_)
    # index for Real
    idx_real = cls_order.index("Real")
    auc = roc_auc_score(y_true, y_proba[:, idx_real])
    print(f"\nLogisticRegression ROC-AUC (Real as positive): {auc:.4f}")
except Exception as e:
    print(f"\n(AUC skipped) Reason: {e}")

# ---------------------------------
# 7) Save vectorizer & models
# ---------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

for key, path in MODEL_PATHS.items():
    with open(path, "wb") as f:
        pickle.dump(models[key], f)

print(f"\n✅ Training complete. Saved to '{SAVE_DIR}/'")
print("   - vectorizer.pkl")
for k, p in MODEL_PATHS.items():
    print(f"   - {os.path.basename(p)}")
