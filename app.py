from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # This renders the HTML form (index.html)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        scaled_features = scaler.transform(features)

        cluster = kmeans.predict(scaled_features)[0]

        return render_template('index.html', predicted_cluster=cluster)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
