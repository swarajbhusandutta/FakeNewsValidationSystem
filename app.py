from flask import Flask, render_template, request, jsonify
from core.scrape import scrape_content
from core.search import search_news
from core.reputation import calculate_reputation
from core.adversarial import detect_adversarial_samples
from core.ml_model import predict_fake_news
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the frontend."""
    try:
        # Get input data from the request
        data = request.json
        input_type = data['type']
        input_data = data['data']

        # Step 1: Detect adversarial patterns
        adversarial_result = detect_adversarial_samples(input_data)
        if adversarial_result["is_adversarial"]:
            return jsonify({
                "error": "Adversarial input detected",
                "flags": adversarial_result["adversarial_flags"],
                "cleaned_text": adversarial_result["cleaned_text"]
            }), 400  # Return error with flags and cleaned text

        cleaned_text = adversarial_result["cleaned_text"]

        # Step 2: Process the input (URL or text)
        if input_type == "url":
            content = scrape_content(cleaned_text)
        else:
            content = cleaned_text

        # Step 3: Perform search and calculate reputation
        search_results = search_news(content)
        ranked_domains, reputation_score = calculate_reputation(search_results, "data/reputation.csv")

        # Step 4: Predict fake news scores using ML models
        predictions, best_model = predict_fake_news(content, reputation_score)

        # Convert predictions and reputation_score to standard Python types
        predictions = {model: float(score) for model, score in predictions.items()}
        reputation_score = float(reputation_score)

        # Step 5: Prepare the response
        return jsonify({
            "content": content,
            "adversarial_flag": False,
            "predictions": predictions,
            "best_model": best_model,
            "search_results": search_results,
            "reputation_score": reputation_score,
            "ranked_domains": ranked_domains
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
