"""
visualize.py
------------
Generates ROC curves, learning curves, and precision-recall curves
for all trained models. Saves all plots to outputs/plots/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve

PLOT_DIR = "outputs/plots"


def _ensure_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_roc_curves(models: dict, X_test, y_test, label_map: dict):
    """
    Plots one-vs-rest ROC curve per class for each model.
    models : {name: fitted_model}
    """
    _ensure_dir()
    classes = sorted(label_map.keys())
    n_classes = len(classes)
    y_bin = label_binarize(y_test, classes=classes)

    for name, model in models.items():
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = cm.tab10(np.linspace(0, 1, n_classes))

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        else:
            # Logistic regression with decision_function fallback
            y_score = model.decision_function(X_test)
            # Normalize to [0,1] per class
            y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{label_map[i]} (AUC = {roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title(f"ROC Curves — {name}", fontsize=15, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

        path = os.path.join(PLOT_DIR, f"roc_{name.replace(' ', '_').lower()}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[PLOT] Saved ROC curve → {path}")


def plot_precision_recall(models: dict, X_test, y_test, label_map: dict):
    """
    Plots precision-recall curves per class for each model.
    """
    _ensure_dir()
    classes = sorted(label_map.keys())
    n_classes = len(classes)
    y_bin = label_binarize(y_test, classes=classes)

    for name, model in models.items():
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = cm.Set2(np.linspace(0, 1, n_classes))

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        else:
            y_score = model.decision_function(X_test)
            y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

        for i, color in zip(range(n_classes), colors):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
            ap = average_precision_score(y_bin[:, i], y_score[:, i])
            ax.plot(recall, precision, color=color, lw=2,
                    label=f"{label_map[i]} (AP = {ap:.2f})")

        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.set_title(f"Precision-Recall Curves — {name}", fontsize=15, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

        path = os.path.join(PLOT_DIR, f"pr_{name.replace(' ', '_').lower()}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[PLOT] Saved PR curve → {path}")


def plot_learning_curves(models: dict, X_train, y_train):
    """
    Plots learning curves (train vs. validation accuracy vs. training size).
    Skips Random Forest to keep runtime reasonable (uses 50 estimators for speed).
    """
    _ensure_dir()
    train_sizes = np.linspace(0.1, 1.0, 8)

    for name, model in models.items():
        print(f"[PLOT] Computing learning curve for {name}...")
        train_sz, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=5, scoring="f1_weighted",
            n_jobs=-1, verbose=0
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.fill_between(train_sz, train_mean - train_std, train_mean + train_std, alpha=0.15, color="royalblue")
        ax.fill_between(train_sz, val_mean - val_std, val_mean + val_std, alpha=0.15, color="tomato")
        ax.plot(train_sz, train_mean, "o-", color="royalblue", lw=2, label="Training F1")
        ax.plot(train_sz, val_mean,   "s-", color="tomato",    lw=2, label="Validation F1")

        ax.set_xlabel("Training Examples", fontsize=13)
        ax.set_ylabel("Weighted F1 Score", fontsize=13)
        ax.set_title(f"Learning Curve — {name}", fontsize=15, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.05])

        path = os.path.join(PLOT_DIR, f"lc_{name.replace(' ', '_').lower()}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[PLOT] Saved learning curve → {path}")


def plot_model_comparison(results: dict):
    """
    Bar chart comparing accuracy and weighted F1 across all models.
    """
    _ensure_dir()
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    f1s   = [results[n]["f1_weighted"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="steelblue")
    bars2 = ax.bar(x + width/2, f1s,  width, label="Weighted F1", color="coral")

    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Comparison — Accuracy vs. Weighted F1", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    path = os.path.join(PLOT_DIR, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved model comparison → {path}")
