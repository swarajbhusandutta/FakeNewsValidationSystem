import joblib
import pandas as pd

# Load models and vectorizer
xgboost_model = joblib.load("models/xgboost_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def predict_fake_news(content, reputation_score):
    """
    Predict fake news probability using all models.
    """
    # Step 1: Transform content into TF-IDF features
    tfidf_features = vectorizer.transform([content])
    reputation_vector = pd.DataFrame([reputation_score], columns=["reputation"])

    # Convert TF-IDF features to DataFrame
    tfidf_df = pd.DataFrame(tfidf_features.toarray())

    # Combine TF-IDF features with the reputation vector
    features = pd.concat([tfidf_df, reputation_vector], axis=1)

    # Ensure all column names are strings
    features.columns = features.columns.astype(str)

    # Step 2: Make predictions with each model
    predictions = {
        "XGBoost": xgboost_model.predict_proba(features)[0][1],  # Probability of being fake news
        "Random Forest": rf_model.predict_proba(features)[0][1]
    }

    # Step 3: Determine the best model based on predictions
    best_model = max(predictions, key=predictions.get)

    return predictions, best_model
