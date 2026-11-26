"""
Intelligent Email/SMS Spam Detection System - CLI Runner
Allows running predictions across all 4 algorithms (Naive Bayes, SVM, Logistic Regression, Decision Tree)
directly from the terminal.
"""

import os
import sys
import pickle
import string
import argparse

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DIR = os.path.join(BASE_DIR, "nltk_data")
if os.path.exists(NLTK_DIR) and NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DIR)

lemma = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

def transform_text(text):
    text = str(text).lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ''.join(ch if ch.isalpha() or ch.isspace() else ' ' for ch in text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    tokens = [lemma.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def load_resources():
    vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
    if not os.path.exists(vectorizer_path):
        vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

    with open(vectorizer_path, "rb") as f:
        tfidf = pickle.load(f)

    models = {}
    models_dir = os.path.join(BASE_DIR, "models")
    algo_map = {
        "Naive Bayes (Bernoulli)": "naive_bayes_bernoullinb.pkl",
        "Naive Bayes (Multinomial)": "naive_bayes_multinomialnb.pkl",
        "Support Vector Machine (SVM)": "support_vector_machine_svm.pkl",
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl"
    }

    for name, fn in algo_map.items():
        fp = os.path.join(models_dir, fn)
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                models[name] = pickle.load(f)

    # Fallback to model.pkl if models dir is empty
    if not models and os.path.exists(os.path.join(BASE_DIR, "model.pkl")):
        with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
            models["Naive Bayes (Default)"] = pickle.load(f)

    return tfidf, models

def predict_message(message, tfidf, models):
    cleaned = transform_text(message)
    vector = tfidf.transform([cleaned]).toarray()
    
    print("\n" + "=" * 70)
    print("INPUT MESSAGE:")
    print(f"  \"{message}\"")
    print("-" * 70)
    print("PREPROCESSED TOKENS:")
    print(f"  \"{cleaned}\"")
    print("-" * 70)
    print(f"{'ALGORITHM':<30} | {'PREDICTION':<10} | {'CONFIDENCE':<12}")
    print("-" * 70)

    for name, model in models.items():
        pred = model.predict(vector)[0]
        label = "🚨 SPAM" if pred == 1 else "✅ HAM"
        conf_str = "N/A"
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(vector)[0]
                conf = prob[pred] * 100
                conf_str = f"{conf:.1f}%"
            except Exception:
                pass
        print(f"{name:<30} | {label:<10} | {conf_str:<12}")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Classify Email/SMS messages as Spam or Ham.")
    parser.add_argument("--message", "-m", type=str, help="Text message to classify", default=None)
    args = parser.parse_args()

    tfidf, models = load_resources()

    if args.message:
        predict_message(args.message, tfidf, models)
    else:
        # Run standard test suite
        print("Running verification test suite on standard test messages:")
        test_messages = [
            "WINNER!! As a valued customer you have won a £900 prize reward! Call 09061701461 to claim now. Claim code KL341.",
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt CLAIM to 81010 now!",
            "Hey, are we still meeting for lunch at 12:30 pm today near the central library?",
            "I have uploaded the latest project report to the shared drive. Please review it when you have time."
        ]
        for msg in test_messages:
            predict_message(msg, tfidf, models)

if __name__ == "__main__":
    main()
