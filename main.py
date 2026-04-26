"""
main.py
-------
Entry point for the Mental Health NLP Classification project.

Usage:
    python main.py --data data/mental_health_reddit.csv

Dataset:
    Reddit Mental Health Dataset from Kaggle:
    https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data
    Expected columns: 'post' (text), 'label' (condition category)

Pipeline:
    1. Load & clean raw Reddit posts
    2. Tokenize, remove stopwords, lemmatize (NLTK + spaCy)
    3. TF-IDF vectorization (unigrams + bigrams, 10k features)
    4. Train Naive Bayes, Logistic Regression, Random Forest
    5. Evaluate with accuracy, F1, classification report
    6. Plot ROC curves, PR curves, learning curves, model comparison
    7. Save best model + vectorizer to disk
"""

import os
import pickle
import argparse
from utils.preprocess import load_and_prepare
from models.train import train_and_evaluate, MODELS
from utils.visualize import (
    plot_roc_curves,
    plot_precision_recall,
    plot_learning_curves,
    plot_model_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Mental Health NLP Classifier")
    parser.add_argument("--data", type=str, default="data/mental_health_reddit.csv",
                        help="Path to the Reddit Mental Health CSV dataset")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use for testing (default: 0.2)")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Skip generating plots (faster run)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  NLP Program for Mental Health Diagnosis")
    print("  Reddit Mental Health Dataset | NLTK + spaCy + sklearn")
    print("="*60 + "\n")

    # Step 1: Load and preprocess
    X_train, X_test, y_train, y_test, vectorizer, label_map = load_and_prepare(
        csv_path=args.data,
        text_col="post",
        label_col="label",
        test_size=args.test_size,
    )

    # Save vectorizer and label map for inference
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("outputs/label_map.pkl", "wb") as f:
        pickle.dump(label_map, f)
    print("[INFO] Vectorizer and label map saved to outputs/\n")

    # Step 2: Train and evaluate all models
    results = train_and_evaluate(X_train, X_test, y_train, y_test, label_map)

    # Step 3: Visualizations
    if not args.skip_plots:
        print("\n[INFO] Generating visualizations...")

        # Build fitted model dict for plotting
        fitted_models = {}
        for name, model in MODELS.items():
            model.fit(X_train, y_train)   # already fitted in train_and_evaluate; re-fit is fast
            fitted_models[name] = model

        plot_roc_curves(fitted_models, X_test, y_test, label_map)
        plot_precision_recall(fitted_models, X_test, y_test, label_map)
        plot_learning_curves(fitted_models, X_train, y_train)
        plot_model_comparison(results)
        print("\n[INFO] All plots saved to outputs/plots/")

    # Step 4: Summary
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    for name, res in results.items():
        print(f"  {name:<25} Acc: {res['accuracy']:.4f}  F1: {res['f1_weighted']:.4f}")
    print("\n[DONE] Run python inference.py \"<your text here>\" to test predictions.\n")


if __name__ == "__main__":
    main()
