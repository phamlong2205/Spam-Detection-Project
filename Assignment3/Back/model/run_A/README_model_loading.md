
# SMS Spam Detection Models - Loading Instructions
# Generated: 20250918_105951

## Quick Start - Load Best Performing Model

```python
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# For LSTM (Best Overall Performance)
# Load LSTM model and preprocessors
lstm_model = load_model('saved_models_20250918_105951/lstm_model.h5')
with open('saved_models_20250918_105951/lstm_tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
label_encoder = joblib.load('saved_models_20250918_105951/label_encoder.joblib')

# For Classical Models (Faster Inference)
# Load SVM/Random Forest and preprocessors
svm_model = joblib.load('saved_models_20250918_105951/svm_model.joblib')
rf_model = joblib.load('saved_models_20250918_105951/random_forest_model.joblib')
tfidf_vectorizer = joblib.load('saved_models_20250918_105951/tfidf_vectorizer.joblib')
scaler = joblib.load('saved_models_20250918_105951/standard_scaler.joblib')
```

## Model Performance Summary
- LSTM: 98.03% accuracy, 92.47% F1-score (Best Overall)
- SVM: 97.94% accuracy, 92.26% F1-score (Fast & Reliable)
- Random Forest: 97.85% accuracy, 91.61% F1-score (Most Interpretable)

## Prediction Example

```python
def predict_spam_lstm(message):
    # Preprocess message for LSTM
    sequence = tokenizer.texts_to_sequences([message])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=21
    )
    
    # Get prediction
    prediction = lstm_model.predict(padded)[0][0]
    label = 'spam' if prediction > 0.5 else 'ham'
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return label, confidence

# Example usage
message = "URGENT! You've won $1000! Click here now!"
label, confidence = predict_spam_lstm(message)
print(f"Prediction: {label} ({confidence:.2%} confidence)")
```

## Files Included
- lstm_model.h5: LSTM neural network model
- svm_model.joblib: Support Vector Machine model  
- random_forest_model.joblib: Random Forest model
- tfidf_vectorizer.joblib: TF-IDF text vectorizer
- standard_scaler.joblib: Numerical feature scaler
- lstm_tokenizer.pickle: LSTM text tokenizer
- label_encoder.joblib: Label encoder (ham/spam)
- model_metadata.json: Complete model configuration
- model_comparison_results.csv: Performance metrics
