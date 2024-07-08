from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Add this line to handle CORS

# Debugging: Print the current working directory
print("Current working directory:", os.getcwd())

# Provide the absolute path to the model file
model_path = './crop_prediction_model.pkl'

# Check if the model file exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the trained model
try:
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Debugging: Print the received input data
        print("Received input data:", data)

        # Validate and prepare input data for prediction
        input_data = [
            float(data['Nitrogen']),
            float(data['Phosphorus']),
            float(data['Potassium']),
            float(data['Temperature']),
            float(data['Humidity']),
            float(data['ph']),
            float(data['Rainfall'])
        ]

        # Debugging: Print the prepared input data
        print("Prepared input data for prediction:", input_data)

        # Make prediction
        prediction = model.predict([input_data])

        # Debugging: Print the prediction result
        print("Prediction result:", prediction)

        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        # Handle any exceptions that occur during the prediction process
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred while making the prediction. Please try again.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
