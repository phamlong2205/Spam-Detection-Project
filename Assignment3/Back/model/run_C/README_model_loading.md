
# Spam Detection Models - Loading Instructions
# Generated: 20251007_201804

## Quick Start - Load Best Performing Model

```python
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# Load LSTM model and preprocessors
lstm_model = load_model('saved_models_20251007_201804/lstm_model.h5')
with open('saved_models_20251007_201804/lstm_tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
label_encoder = joblib.load('saved_models_20251007_201804/label_encoder.joblib')

# Load classical models and preprocessors
random_forest_model = joblib.load('saved_models_20251007_201804/random_forest_model.joblib')
svm_model = joblib.load('saved_models_20251007_201804/svm_model.joblib')
tfidf_vectorizer = joblib.load('saved_models_20251007_201804/tfidf_vectorizer.joblib')
scaler = joblib.load('saved_models_20251007_201804/standard_scaler.joblib')
```

## Focused Feature Set Used
- **Text Features**: cleaned_message (TF-IDF → 800 selected features)
- **Numerical Features**: message_length, digit_ratio, capital_ratio, special_char_count, url_count
- **Boolean Features**: subject_has_suspicious_words

## Prediction Example

```python
def predict_spam_lstm(message):
    # Preprocess message for LSTM
    sequence = tokenizer.texts_to_sequences([message])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=143
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
- random_forest_model.joblib: Random Forest model
- svm_model.joblib: Support Vector Machine model  
- tfidf_vectorizer.joblib: TF-IDF text vectorizer
- feature_selector.joblib: Feature selection transformer
- standard_scaler.joblib: Numerical feature scaler
- lstm_tokenizer.pickle: LSTM text tokenizer
- label_encoder.joblib: Label encoder (ham/spam)
- model_metadata.json: Complete model configuration
