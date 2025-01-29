import joblib

def test_model_loading():
    model = joblib.load("models/xgboost_model.pkl")
    assert model is not None

def test_vectorizer_loading():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    assert vectorizer is not None
