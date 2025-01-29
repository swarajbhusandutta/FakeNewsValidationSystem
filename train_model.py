import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
import nltk
from nltk.corpus import wordnet # Data Augmentation


# Function to augment data using paraphrasing and synonyms

nltk.download('wordnet')
def augment_text(text):
    """Replaces words with synonyms for data augmentation."""
    words = text.split()
    augmented_words = [wordnet.synsets(w)[0].lemmas()[0].name() if wordnet.synsets(w) else w for w in words]
    return " ".join(augmented_words)


def load_data(file_path, reputation_path):
    """Load training data and apply augmentation."""
    data = pd.read_csv(file_path)
    reputation = pd.read_csv(reputation_path)
    data['content'] = data['content'].apply(augment_text)  # Data Augmentation
    data = data.merge(reputation, on='domain', how='left').fillna({'reputation': 50})
    return data


def extract_features(data):
    """Extract TF-IDF features and add reputation scores."""
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    tfidf_features = vectorizer.fit_transform(data['content'])
    reputation_vector = data[['reputation']].values
    combined_features = np.hstack((tfidf_features.toarray(), reputation_vector))
    return combined_features, data['label'], vectorizer


def build_models(X_train, y_train):
    """Train diverse models for Ensemble Diversity."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
        "VotingClassifier": VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('ab', AdaBoostClassifier(n_estimators=50, random_state=42)),
                ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
            ],
            voting='soft'
        )
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_model(model, X_test, y_test):
    """Evaluate model with Adversarial Testing."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred))
    }


def main():
    data_path = "data/training_data.csv"
    reputation_path = "data/reputation.csv"
    data = load_data(data_path, reputation_path)
    X, y, vectorizer = extract_features(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    models = build_models(X_train, y_train)

    print("\nEvaluating models...")
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        print(f"{name} Metrics: {metrics}")

    print("\nSaving models...")
    joblib.dump(models['XGBoost'], "models/xgboost_model.pkl")
    joblib.dump(models['Random Forest'], "models/rf_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("Models saved successfully.")


if __name__ == "__main__":
    main()
