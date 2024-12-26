from flask import Flask, request, jsonify
import pickle

with open('model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')

    X = vectorizer.transform([message])
    prediction = model.predict(X)[0]

    return jsonify({'isOffensive': bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)