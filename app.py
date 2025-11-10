from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.join(BASE_DIR, '..', 'ai')

model_path = os.path.join(AI_DIR, 'spam_model.pkl')
vectorizer_path = os.path.join(AI_DIR, 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')

    if not message.strip():
        return jsonify({'error': '메시지를 입력하세요.'}), 400

    message_tfidf = vectorizer.transform([message])

    prediction = model.predict(message_tfidf)[0]
    probability = model.predict_proba(message_tfidf)[0][1]

    return jsonify({
        'message': message,
        'spam_probability': round(probability * 100, 2),
        'prediction': '스팸' if prediction == 1 else '정상'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
