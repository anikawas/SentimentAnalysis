from flask import Flask, request, jsonify
from src.predict import SentimentPredictor

app = Flask(__name__)
predictor = SentimentPredictor('models/bert_sentiment_model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    sentiment = predictor.predict(data)
    return jsonify({'sentiment': sentiment})

if __name__ == "__main__":
    app.run()
