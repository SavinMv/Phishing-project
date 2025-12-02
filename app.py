# app.py
# -----------------------------
# Phishing URL detector (Live data)
# - PhishTank (phishing) + Tranco (legit) dataset builder
# - TF-IDF (char n-grams) features
# - RandomForest + XGBoost models with quick metrics
# - Streamlit UI for training + prediction
# -----------------------------

import os
import io
import re
import zipfile
import pickle
import datetime as dt
from urllib.parse import urlparse

import requests
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from xgboost import XGBClassifier

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Phishing Detector (PhishTank + Tranco)", layout="wide")
st.title("🛡️ Phishing Website Detection — Live (PhishTank + Tranco)")

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model_v1.pkl")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------- Helpers ----------
PHISHTANK_JSON = "http://data.phishtank.com/data/online-valid.json"
TRANCO_ZIP = "https://tranco-list.eu/top-1m.csv.zip"

def normalize_url(u: str) -> str:
    u = u.strip()
    if not u:
        return u
    # Add scheme if missing
    if not re.match(r"^https?://", u, flags=re.I):
        u = "http://" + u
    # Lowercase hostname only
    try:
        p = urlparse(u)
        host = (p.hostname or "").lower()
        # Rebuild without changing path/query
        scheme = p.scheme if p.scheme else "http"
        netloc = host
        if p.port:
            netloc = f"{host}:{p.port}"
        return f"{scheme}://{netloc}{p.path or ''}{'?' + p.query if p.query else ''}"
    except Exception:
        return u

@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_phishtank(limit: int = 1000) -> pd.DataFrame:
    """Fetch verified online phishing URLs from PhishTank (JSON)."""
    r = requests.get(PHISHTANK_JSON, timeout=45)
    r.raise_for_status()
    data = r.json()
    urls = []
    times = []
    ids = []
    for entry in data[:limit]:
        urls.append(entry.get("url", ""))
        times.append(entry.get("submission_time", ""))
        ids.append(entry.get("phish_id", ""))
    df = pd.DataFrame({"url": urls, "submission_time": times, "phish_id": ids})
    df["url"] = df["url"].astype(str).map(normalize_url)
    df = df[df["url"].str.len() > 0].drop_duplicates(subset=["url"]).reset_index(drop=True)
    df["label"] = 1  # phishing
    return df

@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_tranco(limit: int = 3000) -> pd.DataFrame:
    """Fetch top legitimate domains from Tranco and convert to https://domain form."""
    r = requests.get(TRANCO_ZIP, timeout=45)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        # Default filename in ZIP is "top-1m.csv"
        with z.open("top-1m.csv") as f:
            top = pd.read_csv(f, header=None, names=["rank", "domain"])
    top = top.head(limit).copy()
    top["domain"] = top["domain"].astype(str).str.strip()
    
    urls = []
    # Add both https:// and https://www. variants for diversity
    for domain in top["domain"]:
        urls.append(f"https://{domain}")
        # Add www variant if not already present
        if not domain.startswith("www."):
            urls.append(f"https://www.{domain}")
    
    df = pd.DataFrame({"url": urls})
    df["url"] = df["url"].map(normalize_url)
    df = df[df["url"].str.len() > 0].drop_duplicates(subset=["url"]).reset_index(drop=True)
    df["label"] = 0  # legitimate
    return df[["url", "label"]]

def build_dataset(phish_limit: int, safe_limit: int, balance: bool = True) -> pd.DataFrame:
    """Combine phishing + legit URLs, optionally balance classes by downsampling majority."""
    phish = fetch_phishtank(limit=phish_limit)[["url", "label"]]
    legit = fetch_tranco(limit=safe_limit)[["url", "label"]]

    df = pd.concat([phish, legit], ignore_index=True)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    if balance:
        n_phish = (df["label"] == 1).sum()
        n_legit = (df["label"] == 0).sum()
        if n_phish == 0 or n_legit == 0:
            return df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        minority = min(n_phish, n_legit)
        ph = df[df["label"] == 1].sample(n=minority, random_state=42)
        lg = df[df["label"] == 0].sample(n=minority, random_state=42)
        df = pd.concat([ph, lg], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    return df

def train_models(df: pd.DataFrame, random_state: int = 42):
    """Train TF-IDF (char n-grams) + RF/XGB; return artifacts + quick metrics."""
    # Character n-grams work well for URL patterns
    vect = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
    X = vect.fit_transform(df["url"])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state, class_weight="balanced_subsample"
    )
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    models = {"RandomForest": rf, "XGBoost": xgb}

    # Quick metrics
    metrics = {}
    for name, mdl in models.items():
        y_pred = mdl.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        metrics[name] = {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}

    return models, vect, metrics

def save_artifacts(models, vect, extra_meta: dict = None):
    payload = {
        "models": models,
        "vectorizer": vect,
        "saved_at": dt.datetime.utcnow().isoformat() + "Z",
        "meta": extra_meta or {},
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("⚙️ Settings")
    phish_limit = st.slider("PhishTank sample size", 200, 5000, 1000, 100)
    safe_limit = st.slider("Tranco (legit) sample size", 500, 10000, 2000, 100)
    balance = st.toggle("Balance classes (downsample majority)", value=True)
    retrain = st.button("🔁 Refresh data & Retrain")

# ---------- Train / Load ----------
models = None
vect = None
metrics = None
meta_info = {}

if retrain or (not os.path.exists(MODEL_PATH)):
    with st.spinner("Building dataset and training models…"):
        try:
            df = build_dataset(phish_limit=phish_limit, safe_limit=safe_limit, balance=balance)
            meta_info = {
                "num_records": len(df),
                "num_phishing": int((df["label"] == 1).sum()),
                "num_legit": int((df["label"] == 0).sum()),
                "phish_limit": phish_limit,
                "safe_limit": safe_limit,
                "balanced": balance,
            }
            models, vect, metrics = train_models(df)
            save_artifacts(models, vect, extra_meta=meta_info)
            st.success("Training complete and artifacts saved ✅")
        except Exception as e:
            st.error(f"Training failed: {e}")
else:
    try:
        payload = load_artifacts()
        models = payload["models"]
        vect = payload["vectorizer"]
        
        # Validate that models are actually fitted (handles version mismatch issues)
        check_is_fitted(vect)
        for m in models.values():
            check_is_fitted(m)
            
        metrics = payload.get("meta_metrics")
        meta_info = payload.get("meta", {})
        st.info("Loaded existing model artifacts from disk.")
    except Exception as e:
        st.warning(f"Could not load saved model: {e}. Training a fresh model…")
        df = build_dataset(phish_limit=phish_limit, safe_limit=safe_limit, balance=balance)
        models, vect, metrics = train_models(df)
        save_artifacts(models, vect, extra_meta={"num_records": len(df)})

# ---------- Show dataset summary / metrics ----------
cols = st.columns(3)
with cols[0]:
    st.metric("Total URLs used", value=f'{meta_info.get("num_records", "—")}')
with cols[1]:
    st.metric("Phishing (1)", value=f'{meta_info.get("num_phishing", "—")}')
with cols[2]:
    st.metric("Legitimate (0)", value=f'{meta_info.get("num_legit", "—")}')

if metrics:
    st.subheader("📊 Validation Metrics (hold-out test split)")
    mdf = (
        pd.DataFrame(metrics)
        .T[["accuracy", "precision", "recall", "f1"]]
        .sort_values("f1", ascending=False)
        .round(3)
    )
    st.dataframe(mdf, use_container_width=True)

# ---------- Prediction UI ----------
st.subheader("🔎 Check a URL")
model_choice = st.selectbox("Choose a model:", list(models.keys()) if models else ["RandomForest"])
url_input = st.text_input("Enter a website URL (e.g., https://securebank.com/login)")

btn_cols = st.columns([1, 1, 6])
with btn_cols[0]:
    predict_clicked = st.button("Predict")
with btn_cols[1]:
    clear_clicked = st.button("Clear")

if clear_clicked:
    st.rerun()

if predict_clicked and url_input.strip():
    url_norm = normalize_url(url_input)
    if models and vect:
        feats = vect.transform([url_norm])
        mdl = models.get(model_choice)
        pred = mdl.predict(feats)[0]
        proba = None
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(feats)[0, 1]
        label = "🚨 Likely PHISHING" if pred == 1 else "✅ Likely LEGITIMATE"
        st.write(f"**URL:** `{url_norm}`")
        st.markdown(f"**Prediction:** {label}")
        if proba is not None:
            st.progress(float(proba))
            st.caption(f"Model confidence (phishing class): {proba:.3f}")
    else:
        st.error("Model not available. Please retrain.")

# ---------- Notes ----------
with st.expander("ℹ️ Notes & Tips"):
    st.markdown(
        """
- **Data sources**:
  - Phishing: PhishTank verified online feed.
  - Legitimate: Tranco top domains (treated as benign).
- **Features**: TF-IDF on URL character n-grams (3–5). This works well for URL patterns (e.g., suspicious tokens, subdomain tricks).
- **Caveats**:
  - Some top domains may host user content; treating them all as benign is a heuristic.
  - For production, augment with HTML/WHOIS/SSL features and continual retraining.
- **Refresh**: Use *Refresh data & Retrain* in the sidebar to pull the latest feeds and rebuild the model.
        """
    )
