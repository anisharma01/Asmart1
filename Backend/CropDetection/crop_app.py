from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Construct the path to the model file
current_directory = os.path.dirname(__file__)
model_path = os.path.join(current_directory, 'crop_app.joblib')

# Load the trained model
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array([[data['Nitrogen'], data['Phosphorus'], data['Potassium'],
                              data['Temperature'], data['Humidity'], data['ph'], data['Rainfall']]])
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        # Log the error stack trace
        error_message = str(e)
        traceback.print_exc()
        return jsonify({'error': error_message})

if __name__ == '__main__':
    app.run(debug=True)
