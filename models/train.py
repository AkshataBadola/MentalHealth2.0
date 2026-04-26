"""
train.py
--------
Trains Naive Bayes, Logistic Regression, and Random Forest classifiers
on the preprocessed Reddit Mental Health dataset. Saves best model to disk.
"""

import os
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, roc_auc_score
)


MODELS = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        multi_class="multinomial", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_split=5, random_state=42, n_jobs=-1
    ),
}


def train_and_evaluate(X_train, X_test, y_train, y_test, label_map: dict,
                       output_dir: str = "outputs") -> dict:
    """
    Trains all models, prints evaluation metrics, saves best model.

    Parameters
    ----------
    X_train, X_test : sparse matrices
    y_train, y_test : integer label arrays
    label_map       : {int: label_string}
    output_dir      : where to save model pickle

    Returns
    -------
    results : dict of {model_name: {accuracy, f1, report}}
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    best_f1 = 0.0
    best_model_name = None
    best_model_obj = None

    target_names = [label_map[i] for i in sorted(label_map.keys())]

    for name, model in MODELS.items():
        print(f"\n{'='*55}")
        print(f"  Training: {name}")
        print(f"{'='*55}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")

        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        results[name] = {
            "accuracy": acc,
            "f1_weighted": f1,
            "report": classification_report(
                y_test, y_pred, target_names=target_names, output_dict=True
            ),
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    # Save best model
    model_path = os.path.join(output_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model_obj, "name": best_model_name}, f)
    print(f"\n[INFO] Best model: {best_model_name} (F1={best_f1:.4f})")
    print(f"[INFO] Saved to: {model_path}")

    return results
