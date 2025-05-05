from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and features
model = load_model('fraud_detection_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Check for missing features
        missing_features = [f for f in features if f not in input_df.columns]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400
        
        # Select and scale features
        X = input_df[features]
        X_scaled = scaler.transform(X)
        
        # Predict
        fraud_proba = model.predict(X_scaled)[0][0]
        fraud_label = int(fraud_proba > 0.5)
        
        # Return result
        return jsonify({
            'fraud_probability': float(fraud_proba),
            'fraud_label': fraud_label
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)