"""
preprocess.py
-------------
Text cleaning and feature extraction pipeline for mental health classification.
Uses NLTK and spaCy for tokenization, stopword removal, lemmatization,
and TF-IDF vectorization.
"""

import re
import nltk
import spacy
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download required NLTK resources on first run
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Lowercases, removes URLs, special characters, digits, and extra whitespace.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # keep only letters
    text = re.sub(r"\s+", " ", text).strip()             # collapse whitespace
    return text


def tokenize_and_lemmatize(text: str) -> str:
    """
    Tokenizes with NLTK, removes stopwords, then lemmatizes with spaCy.
    Returns a cleaned string of lemmas.
    """
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(lemmas)


def full_pipeline(text: str) -> str:
    """End-to-end: clean → tokenize → lemmatize."""
    return tokenize_and_lemmatize(clean_text(text))


def load_and_prepare(csv_path: str, text_col: str = "post", label_col: str = "label",
                     test_size: float = 0.2, random_state: int = 42):
    """
    Loads the Reddit Mental Health CSV, applies full pipeline,
    vectorizes with TF-IDF, and returns train/test splits.

    Expected CSV columns:
        - post  : raw Reddit post text
        - label : mental health category string

    Returns
    -------
    X_train, X_test : scipy sparse matrices (TF-IDF features)
    y_train, y_test : numpy arrays (integer-encoded labels)
    vectorizer      : fitted TfidfVectorizer (for inference)
    label_map       : dict mapping integer → label string
    """
    print(f"[INFO] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Drop rows with missing values
    df = df[[text_col, label_col]].dropna().reset_index(drop=True)
    print(f"[INFO] Loaded {len(df)} samples across {df[label_col].nunique()} classes")
    print(f"[INFO] Class distribution:\n{df[label_col].value_counts()}\n")

    # Apply preprocessing pipeline
    print("[INFO] Running preprocessing pipeline (clean → tokenize → lemmatize)...")
    df["processed"] = df[text_col].apply(full_pipeline)

    # Encode labels
    labels = df[label_col].unique()
    label_map = {i: lab for i, lab in enumerate(sorted(labels))}
    label_map_inv = {v: k for k, v in label_map.items()}
    df["label_enc"] = df[label_col].map(label_map_inv)

    # TF-IDF vectorization
    print("[INFO] Fitting TF-IDF vectorizer (max_features=10000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),       # unigrams + bigrams
        sublinear_tf=True,        # log-scaled TF
        min_df=3,                 # ignore very rare terms
        max_df=0.90,              # ignore very common terms
    )
    X = vectorizer.fit_transform(df["processed"])
    y = df["label_enc"].values

    # Stratified split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[INFO] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, vectorizer, label_map
