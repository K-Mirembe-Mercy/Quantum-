"""
Text preprocessing + feature extraction for the quantum classifier.

Pipeline:
  raw text → clean → TF-IDF (vocab 500) → PCA (4 dims) → angle-scale to [0, π]
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ── Built-in demo dataset ────────────────────────────────────────────────────

DEMO_SENTENCES = [
    # Positive
    ("This product is absolutely amazing and works perfectly!", 1),
    ("I love this so much, best purchase I've ever made.", 1),
    ("Fantastic quality, fast shipping, highly recommend!", 1),
    ("Exceeded my expectations. Will definitely buy again.", 1),
    ("Great value for money. Very happy with this.", 1),
    ("Outstanding performance, couldn't be more satisfied.", 1),
    ("Brilliant design, easy to use and looks great.", 1),
    ("Five stars! This changed my life for the better.", 1),
    ("So happy with this purchase, works like a charm.", 1),
    ("Incredible product, my whole family loves it.", 1),
    ("The quality is top-notch and it arrived quickly.", 1),
    ("Perfect in every way, exactly as described.", 1),
    ("Really impressed, does everything it claims to do.", 1),
    ("Best thing I've bought all year, truly wonderful.", 1),
    ("Super fast delivery and the item is fantastic!", 1),
    # Negative
    ("Terrible quality, broke after one day. Avoid!", 0),
    ("Complete waste of money. Does not work at all.", 0),
    ("Very disappointed. Arrived damaged and unusable.", 0),
    ("Worst purchase ever. Nothing like the description.", 0),
    ("Poor craftsmanship, fell apart immediately.", 0),
    ("Do not buy this. Absolute garbage product.", 0),
    ("Extremely frustrated. Customer service was useless.", 0),
    ("Defective item, stopped working within a week.", 0),
    ("Huge disappointment. Expected much better quality.", 0),
    ("Returned immediately. Cheap, flimsy, and overpriced.", 0),
    ("Not what I ordered and the quality is awful.", 0),
    ("Broke on first use. Total waste of time and money.", 0),
    ("Horrible experience from start to finish.", 0),
    ("Absolutely terrible. Would give zero stars if I could.", 0),
    ("Faulty product, non-existent support. Stay away.", 0),
]


# ── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Feature Pipeline ─────────────────────────────────────────────────────────

class QuantumFeaturePipeline:
    """
    Converts raw text into 4-dimensional quantum-ready feature vectors.

    Fit on training data, transform both train and test.
    """

    def __init__(self, n_features: int = 4, vocab_size: int = 500):
        self.n_features = n_features
        self.tfidf = TfidfVectorizer(
            max_features=vocab_size,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.pca   = PCA(n_components=n_features)
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))

    def fit_transform(self, texts):
        cleaned = [clean_text(t) for t in texts]
        tfidf_matrix = self.tfidf.fit_transform(cleaned).toarray()
        pca_out = self.pca.fit_transform(tfidf_matrix)
        scaled  = self.scaler.fit_transform(pca_out)
        return scaled

    def transform(self, texts):
        cleaned = [clean_text(t) for t in texts]
        tfidf_matrix = self.tfidf.transform(cleaned).toarray()
        pca_out = self.pca.transform(tfidf_matrix)
        scaled  = self.scaler.transform(pca_out)
        return scaled

    def transform_single(self, text: str):
        return self.transform([text])[0]


# ── Dataset Helpers ──────────────────────────────────────────────────────────

def load_demo_dataset(test_size=0.25, seed=42):
    """
    Returns train/test splits from the built-in demo dataset.
    Also returns the fitted feature pipeline.
    """
    texts  = [s for s, _ in DEMO_SENTENCES]
    labels = np.array([l for _, l in DEMO_SENTENCES])

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    pipeline = QuantumFeaturePipeline()
    X_tr = pipeline.fit_transform(X_tr_raw)
    X_te = pipeline.transform(X_te_raw)

    return X_tr, X_te, y_tr, y_te, pipeline


def load_custom_dataset(pos_texts, neg_texts, test_size=0.25, seed=42):
    """
    Build a dataset from user-supplied lists of positive/negative strings.
    """
    texts  = pos_texts + neg_texts
    labels = np.array([1] * len(pos_texts) + [0] * len(neg_texts))

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    pipeline = QuantumFeaturePipeline()
    X_tr = pipeline.fit_transform(X_tr_raw)
    X_te = pipeline.transform(X_te_raw)

    return X_tr, X_te, y_tr, y_te, pipeline
