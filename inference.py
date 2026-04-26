"""
inference.py
------------
Load the best saved model and run predictions on new text inputs.
Run: python inference.py "I have been feeling really hopeless and empty lately."
"""

import sys
import pickle
from utils.preprocess import full_pipeline


MODEL_PATH  = "outputs/best_model.pkl"
VECTOR_PATH = "outputs/vectorizer.pkl"
LABEL_PATH  = "outputs/label_map.pkl"


def load_artifacts():
    with open(MODEL_PATH,  "rb") as f: model_data  = pickle.load(f)
    with open(VECTOR_PATH, "rb") as f: vectorizer  = pickle.load(f)
    with open(LABEL_PATH,  "rb") as f: label_map   = pickle.load(f)
    return model_data["model"], model_data["name"], vectorizer, label_map


def predict(text: str) -> dict:
    model, model_name, vectorizer, label_map = load_artifacts()

    processed = full_pipeline(text)
    X = vectorizer.transform([processed])
    pred_int = model.predict(X)[0]
    pred_label = label_map[pred_int]

    result = {
        "input_text":   text,
        "prediction":   pred_label,
        "model_used":   model_name,
    }

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        result["confidence"] = {label_map[i]: round(float(p), 4) for i, p in enumerate(probs)}

    return result


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "I feel completely empty and nothing seems to bring me joy anymore."

    print(f"\n[INPUT] {text}\n")
    result = predict(text)
    print(f"[PREDICTION] {result['prediction']}")
    print(f"[MODEL]      {result['model_used']}")
    if "confidence" in result:
        print("[CONFIDENCE SCORES]")
        for label, prob in sorted(result["confidence"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"  {label:<20} {prob:.4f}  {bar}")
