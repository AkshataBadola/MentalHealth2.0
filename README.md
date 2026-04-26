# NLP Program for Mental Health Diagnosis

Classifies mental health conditions from Reddit posts using classical NLP + ML.
Built with NLTK, spaCy, and scikit-learn.

## Dataset
Reddit Mental Health Dataset (Kaggle):
https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data

Download the CSV and place it at: `data/mental_health_reddit.csv`

Expected columns: `post` (text), `label` (condition)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download spaCy model
python -m spacy download en_core_web_sm
```

## Run

```bash
# Full pipeline: preprocess → train → evaluate → visualize
python main.py --data data/mental_health_reddit.csv

# Skip plots for faster run
python main.py --data data/mental_health_reddit.csv --skip_plots

# Run inference on custom text
python inference.py "I've been feeling completely hopeless and can't get out of bed."
```

## Project Structure

```
nlp_mental_health/
├── main.py              # Entry point
├── inference.py         # Predict on new text
├── requirements.txt
├── data/
│   └── mental_health_reddit.csv   # (download from Kaggle)
├── models/
│   └── train.py         # Model definitions + training loop
├── outputs/
│   ├── best_model.pkl   # Saved best model
│   ├── vectorizer.pkl   # Fitted TF-IDF vectorizer
│   ├── label_map.pkl    # Integer → label mapping
│   └── plots/           # All generated visualizations
└── utils/
    ├── preprocess.py    # Text cleaning, tokenization, lemmatization, TF-IDF
    └── visualize.py     # ROC, PR, learning curves, model comparison
```

## Pipeline

```
Raw Reddit Post
     ↓
Lowercase + Remove URLs/punctuation/digits
     ↓
NLTK Tokenization + Stopword Removal
     ↓
spaCy Lemmatization
     ↓
TF-IDF Vectorization (10k features, unigrams+bigrams)
     ↓
┌──────────────────────────────────┐
│  Naive Bayes  │  LR  │  RF      │
└──────────────────────────────────┘
     ↓
Evaluation: Accuracy, F1, ROC, PR Curves
```
