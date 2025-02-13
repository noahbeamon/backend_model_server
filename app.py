from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from your React frontend

# Load the pre-trained machine learning model
model = joblib.load("./ttm_pue_model.pkl")  # Ensure this path is correct and matches your model file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Extract the required input parameters
        altitude = data.get('altitude')
        avg_temp = data.get('average_temperature')
        avg_humidity = data.get('average_humidity')
        
        # Check if any of the required parameters are missing
        if altitude is None or avg_temp is None or avg_humidity is None:
            return jsonify({'error': 'Missing input parameters'}), 400
        
        # Prepare the input for the model
        input_features = np.array([[altitude, avg_temp, avg_humidity]])
        print(f"Input Features: {input_features}")  # Debugging print statement
        
        # Make the prediction
        prediction = model.predict(input_features)[0]  # Get the first prediction
        print(f"Predicted Trailing twelve-month (TTM) PUE: {prediction}")  # Debugging print statement
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) #run the flask app

