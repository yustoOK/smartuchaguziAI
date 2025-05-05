Smartuchaguzi Fraud Detection API

This project provides a machine learning-powered API to detect fraudulent voting behavior. It uses a pre-trained Keras model to analyze voting session features and return a fraud probability score.

Project Structure:
- Data/       : Contains sample raw data (not full training data due to size)
- Train/      : Includes training scripts and preprocessing code
- api/        : API logic and model files (fraud_api.py, .keras model, scaler, feature list)
- requirements.txt
- render.yaml
- README.txt

How It Works:
1. Receives JSON input via POST request.
2. Validates input and selects expected features.
3. Scales input using pre-trained scaler.
4. Uses the model to predict fraud probability.
5. Returns probability and binary label (1 = fraud, 0 = normal).

Example Input:
{
  "time_diff": 0.3,
  "votes_per_user": 7,
  "avg_time_between_votes": 0.2,
  "vote_frequency": 2.5,
  "vpn_usage": 1,
  "multiple_logins": 2,
  "session_duration": 15.0,
  "location_flag": 1
}

Example Output:
{
  "fraud_probability": 0.87,
  "fraud_label": 1
}

To Run Locally:
1. Install Python dependencies:
   pip install -r requirements.txt

2. Run the API server:
   python api/fraud_api.py

3. Send a test request using curl or Postman:
   curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @example_input.json

Deployment:
This project is configured for free deployment using Render.com. Ensure your GitHub repository is public. The render.yaml file contains deployment instructions Render will follow.

Note:
Training data (1.5 million rows) is not included due to GitHub size limits. Only sample files are provided for demonstration.

License:
This project is open for academic and non-commercial use.
