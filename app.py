"""
Flask Web Application for California Housing Prediction
Provides web interface for ML model predictions
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
from preprocessing import CaliforniaHousingPreprocessor

app = Flask(__name__)

# Global variables for models and preprocessor
preprocessor = None
regression_models = {}
classification_models = {}
neural_model = None
label_encoder = None

def load_models():
    """Load all saved models and preprocessor"""
    global preprocessor, regression_models, classification_models, neural_model, label_encoder
    
    try:
        # Load preprocessor and scaler
        preprocessor = CaliforniaHousingPreprocessor()
        preprocessor.load_scaler('saved_models/scaler.joblib')
        
        # Load regression models
        regression_models['simple_linear'] = joblib.load('saved_models/simple_linear_regression.joblib')
        regression_models['multiple_linear'] = joblib.load('saved_models/multiple_linear_regression.joblib')
        
        # Load classification models
        classification_models['logistic'] = joblib.load('saved_models/logistic_classifier.joblib')
        classification_models['decision_tree'] = joblib.load('saved_models/decision_tree_classifier.joblib')
        classification_models['random_forest'] = joblib.load('saved_models/random_forest_classifier.joblib')
        classification_models['svm_rbf'] = joblib.load('saved_models/svm_rbf_classifier.joblib')
        
        # Load label encoder
        label_encoder = joblib.load('saved_models/label_encoder.joblib')
        
        # Load neural network
        import tensorflow as tf
        neural_model = tf.keras.models.load_model('saved_models/neural_network.h5')
        
        print("All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def get_classification_thresholds():
    """Get classification thresholds for housing values"""
    # These thresholds should match what was used during training
    # For California housing, approximate values based on data distribution
    low_threshold = 1.196  # 33rd percentile
    high_threshold = 2.647  # 67th percentile
    return low_threshold, high_threshold

def predict_regression(features):
    """Make regression predictions"""
    try:
        # Convert to DataFrame
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                        'Population', 'AveOccup', 'Latitude', 'Longitude']
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # Scale features
        input_scaled = preprocessor.scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        # Make predictions
        simple_pred = regression_models['simple_linear'].predict(input_scaled_df[['MedInc']])[0]
        multiple_pred = regression_models['multiple_linear'].predict(input_scaled_df)[0]
        
        return {
            'simple_linear': float(simple_pred),
            'multiple_linear': float(multiple_pred)
        }
    except Exception as e:
        print(f"Regression prediction error: {e}")
        return None

def predict_classification(features):
    """Make classification predictions"""
    try:
        # Convert to DataFrame
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                        'Population', 'AveOccup', 'Latitude', 'Longitude']
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # Scale features
        input_scaled = preprocessor.scaler.transform(input_df)
        
        # Make predictions with different models
        predictions = {}
        
        # Logistic Regression
        log_pred = classification_models['logistic'].predict(input_scaled)[0]
        predictions['logistic'] = label_encoder.inverse_transform([log_pred])[0]
        
        # Decision Tree
        dt_pred = classification_models['decision_tree'].predict(input_scaled)[0]
        predictions['decision_tree'] = label_encoder.inverse_transform([dt_pred])[0]
        
        # Random Forest
        rf_pred = classification_models['random_forest'].predict(input_scaled)[0]
        predictions['random_forest'] = label_encoder.inverse_transform([rf_pred])[0]
        
        # SVM
        svm_pred = classification_models['svm_rbf'].predict(input_scaled)[0]
        predictions['svm_rbf'] = label_encoder.inverse_transform([svm_pred])[0]
        
        # Neural Network
        nn_pred_proba = neural_model.predict(input_scaled)
        nn_pred = np.argmax(nn_pred_proba, axis=1)[0]
        predictions['neural_network'] = label_encoder.inverse_transform([nn_pred])[0]
        
        # Get ensemble prediction (majority vote)
        all_predictions = list(predictions.values())
        ensemble_pred = max(set(all_predictions), key=all_predictions.count)
        predictions['ensemble'] = ensemble_pred
        
        return predictions
        
    except Exception as e:
        print(f"Classification prediction error: {e}")
        return None

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        features = [
            float(request.form['medinc']),
            float(request.form['houseage']),
            float(request.form['averooms']),
            float(request.form['avebedrms']),
            float(request.form['population']),
            float(request.form['aveoccup']),
            float(request.form['latitude']),
            float(request.form['longitude'])
        ]
        
        # Make predictions
        regression_results = predict_regression(features)
        classification_results = predict_classification(features)
        
        if regression_results is None or classification_results is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Prepare response
        response = {
            'regression': regression_results,
            'classification': classification_results,
            'input_features': {
                'MedInc': features[0],
                'HouseAge': features[1],
                'AveRooms': features[2],
                'AveBedrms': features[3],
                'Population': features[4],
                'AveOccup': features[5],
                'Latitude': features[6],
                'Longitude': features[7]
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/model_info')
def model_info():
    """Return model information"""
    info = {
        'regression_models': list(regression_models.keys()),
        'classification_models': list(classification_models.keys()),
        'neural_network': neural_model is not None,
        'feature_names': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                         'Population', 'AveOccup', 'Latitude', 'Longitude'],
        'classification_classes': list(label_encoder.classes_) if label_encoder else []
    }
    return jsonify(info)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_loaded': len(regression_models) > 0})

if __name__ == '__main__':
    print("Starting California Housing ML Web Application...")
    
    # Load models before starting the app
    if load_models():
        print("Models loaded successfully. Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("Failed to load models. Please run the training script first.")
        print("Run: python models.py")
