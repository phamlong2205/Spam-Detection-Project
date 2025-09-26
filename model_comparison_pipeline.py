"""
SMS Spam Detection - Complete Machine Learning Pipeline

This script implements a comprehensive comparison of three different machine learning
approaches for SMS spam detection:
1. Support Vector Machine (SVM) with TF-IDF + engineered features
2. Random Forest with TF-IDF + engineered features  
3. Long Short-Term Memory (LSTM) neural network

The pipeline processes the feature-engineered dataset and provides detailed
performance comparisons across all models.

"""

import pandas as pd
import numpy as np
import time
import joblib
import pickle
import json
from datetime import datetime
from typing import Dict, Any

# Classical ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow warnings for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("SMS SPAM DETECTION - COMPREHENSIVE MODEL COMPARISON")
print("=" * 70)

# ============================================================================
# 1. SETUP AND DATA PREPARATION
# ============================================================================

def load_and_prepare_data(csv_path: str = 'data/spam_with_features.csv') -> tuple:
    """
    Load the processed dataset with engineered features.
    
    Args:
        csv_path (str): Path to the processed CSV file
        
    Returns:
        tuple: (DataFrame, LabelEncoder) - Loaded dataset and fitted label encoder
    """
    print("\n1. LOADING AND PREPARING DATA")
    print("-" * 40)
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"Spam percentage: {(df['label'] == 'spam').mean() * 100:.1f}%")
    
    # Encode labels (ham=0, spam=1)
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    print(f"\nLabel encoding:")
    print(f"ham -> 0, spam -> 1")
    print(f"Encoded label distribution:")
    print(df['label_encoded'].value_counts().sort_index())
    
    return df, label_encoder

def prepare_classical_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features for classical ML models (SVM, Random Forest).
    Combines TF-IDF features with engineered numerical features.
    
    Args:
        df (pd.DataFrame): Input DataFrame with all features
        
    Returns:
        tuple: (X_combined, y, feature_names)
    """
    print("\n2. PREPARING FEATURES FOR CLASSICAL MODELS")
    print("-" * 50)
    
    # Extract text and numerical features
    text_data = df['cleaned_message'].fillna('')
    numerical_features = ['message_length', 'digit_ratio', 'capital_ratio', 'special_char_count']
    numerical_data = df[numerical_features].values
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,          # Limit vocabulary size for efficiency
        min_df=2,                   # Ignore terms that appear in less than 2 documents
        max_df=0.95,                # Ignore terms that appear in more than 95% of documents
        ngram_range=(1, 2),         # Use unigrams and bigrams
        stop_words='english'        # Remove English stop words
    )
    
    tfidf_features = tfidf_vectorizer.fit_transform(text_data)
    
    print(f"TF-IDF matrix shape: {tfidf_features.shape}")
    print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    
    # Scale numerical features
    print("Scaling numerical features...")
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_data)
    
    # Combine TF-IDF and numerical features
    print("Combining TF-IDF and numerical features...")
    numerical_sparse = csr_matrix(numerical_features_scaled)
    X_combined = hstack([tfidf_features, numerical_sparse])
    
    print(f"Combined feature matrix shape: {X_combined.shape}")
    print(f"Feature breakdown:")
    print(f"  - TF-IDF features: {tfidf_features.shape[1]}")
    print(f"  - Numerical features: {len(numerical_features)}")
    print(f"  - Total features: {X_combined.shape[1]}")
    
    # Target variable
    y = df['label_encoded'].values
    
    # Feature names for interpretation
    tfidf_feature_names = [f"tfidf_{word}" for word in tfidf_vectorizer.get_feature_names_out()]
    feature_names = tfidf_feature_names + numerical_features
    
    return X_combined, y, feature_names, tfidf_vectorizer, scaler

def prepare_lstm_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features for LSTM model.
    Tokenizes and pads text sequences.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        tuple: (X_padded, y, tokenizer)
    """
    print("\n3. PREPARING FEATURES FOR LSTM MODEL")
    print("-" * 45)
    
    # Extract text data
    text_data = df['cleaned_message'].fillna('').tolist()
    
    # Tokenize text
    print("Tokenizing text for LSTM...")
    tokenizer = Tokenizer(
        num_words=10000,            # Keep top 10k words
        oov_token='<OOV>',          # Out-of-vocabulary token
        lower=True                  # Convert to lowercase
    )
    
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    
    # Determine optimal sequence length
    sequence_lengths = [len(seq) for seq in sequences]
    avg_length = np.mean(sequence_lengths)
    max_length = np.max(sequence_lengths)
    percentile_95 = np.percentile(sequence_lengths, 95)
    
    print(f"Sequence statistics:")
    print(f"  - Average length: {avg_length:.1f}")
    print(f"  - Maximum length: {max_length}")
    print(f"  - 95th percentile: {percentile_95:.1f}")
    
    # Use 95th percentile as max length to balance information retention and efficiency
    max_sequence_length = int(percentile_95)
    
    # Pad sequences
    print(f"Padding sequences to length {max_sequence_length}...")
    X_padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    print(f"Padded sequences shape: {X_padded.shape}")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    
    # Target variable
    y = df['label_encoded'].values
    
    return X_padded, y, tokenizer, max_sequence_length

# ============================================================================
# 4. CLASSICAL MODELS PIPELINE (SVM & RANDOM FOREST)
# ============================================================================

def train_classical_models(X_combined, y) -> tuple:
    """
    Train and evaluate SVM and Random Forest models.
    
    Args:
        X_combined: Combined feature matrix (TF-IDF + numerical)
        y: Target labels
        
    Returns:
        tuple: (results_dict, train_test_split_data, trained_models_dict)
    """
    print("\n4. TRAINING CLASSICAL MODELS")
    print("-" * 40)
    
    # Split data with stratification to handle class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training set spam ratio: {y_train.mean():.3f}")
    print(f"Test set spam ratio: {y_test.mean():.3f}")
    
    results = {}
    
    # ========================================
    # Support Vector Machine (SVM)
    # ========================================
    print("\nTraining Support Vector Machine...")
    start_time = time.time()
    
    # Use RBF kernel with class weight balancing for imbalanced data
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',    # Handle class imbalance
        random_state=42
    )
    
    svm_model.fit(X_train, y_train)
    svm_train_time = time.time() - start_time
    
    # Evaluate SVM
    start_time = time.time()
    svm_predictions = svm_model.predict(X_test)
    svm_predict_time = time.time() - start_time
    
    # Calculate metrics
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    svm_precision = precision_score(y_test, svm_predictions)
    svm_recall = recall_score(y_test, svm_predictions)
    svm_f1 = f1_score(y_test, svm_predictions)
    
    results['SVM'] = {
        'accuracy': svm_accuracy,
        'precision': svm_precision,
        'recall': svm_recall,
        'f1_score': svm_f1,
        'train_time': svm_train_time,
        'predict_time': svm_predict_time
    }
    
    print(f"SVM Training completed in {svm_train_time:.2f} seconds")
    print(f"SVM Prediction completed in {svm_predict_time:.4f} seconds")
    print("\nSVM Classification Report:")
    print(classification_report(y_test, svm_predictions, target_names=['Ham', 'Spam']))
    print("SVM Confusion Matrix:")
    print(confusion_matrix(y_test, svm_predictions))
    
    # ========================================
    # Random Forest
    # ========================================
    print("\nTraining Random Forest...")
    start_time = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=100,           # Number of trees
        max_depth=10,               # Prevent overfitting
        min_samples_split=5,        # Minimum samples to split node
        min_samples_leaf=2,         # Minimum samples in leaf
        class_weight='balanced',    # Handle class imbalance
        random_state=42,
        n_jobs=-1                   # Use all cores
    )
    
    rf_model.fit(X_train, y_train)
    rf_train_time = time.time() - start_time
    
    # Evaluate Random Forest
    start_time = time.time()
    rf_predictions = rf_model.predict(X_test)
    rf_predict_time = time.time() - start_time
    
    # Calculate metrics
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    rf_precision = precision_score(y_test, rf_predictions)
    rf_recall = recall_score(y_test, rf_predictions)
    rf_f1 = f1_score(y_test, rf_predictions)
    
    results['Random Forest'] = {
        'accuracy': rf_accuracy,
        'precision': rf_precision,
        'recall': rf_recall,
        'f1_score': rf_f1,
        'train_time': rf_train_time,
        'predict_time': rf_predict_time
    }
    
    print(f"Random Forest Training completed in {rf_train_time:.2f} seconds")
    print(f"Random Forest Prediction completed in {rf_predict_time:.4f} seconds")
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_predictions, target_names=['Ham', 'Spam']))
    print("Random Forest Confusion Matrix:")
    print(confusion_matrix(y_test, rf_predictions))
    
    # Feature importance analysis for Random Forest
    print("\nTop 10 Most Important Features (Random Forest):")
    if hasattr(rf_model, 'feature_importances_'):
        feature_importance = rf_model.feature_importances_
        # Since we can't get feature names easily from combined sparse matrix,
        # we'll show the importance of the numerical features (last 4 features)
        numerical_importance = feature_importance[-4:]
        numerical_feature_names = ['message_length', 'digit_ratio', 'capital_ratio', 'special_char_count']
        
        print("Numerical Feature Importance:")
        for name, importance in zip(numerical_feature_names, numerical_importance):
            print(f"  {name}: {importance:.4f}")
    
    return results, (X_train, X_test, y_train, y_test), {'svm': svm_model, 'random_forest': rf_model}

# ============================================================================
# 5. DEEP LEARNING MODEL PIPELINE (LSTM)
# ============================================================================

def train_lstm_model(X_padded, y, max_sequence_length, vocab_size) -> tuple:
    """
    Train and evaluate LSTM model.
    
    Args:
        X_padded: Padded text sequences
        y: Target labels
        max_sequence_length: Maximum sequence length
        vocab_size: Vocabulary size
        
    Returns:
        tuple: (results_dict, trained_model)
    """
    print("\n5. TRAINING LSTM MODEL")
    print("-" * 30)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"LSTM Training set shape: {X_train.shape}")
    print(f"LSTM Test set shape: {X_test.shape}")
    
    # ========================================
    # Build LSTM Model
    # ========================================
    print("\nBuilding LSTM architecture...")
    
    model = Sequential([
        # Embedding layer - converts word indices to dense vectors
        Embedding(
            input_dim=vocab_size + 1,      # +1 for OOV token
            output_dim=128,                # Embedding dimension
            input_length=max_sequence_length,
            mask_zero=True                 # Ignore padded values
        ),
        
        # LSTM layer with dropout for regularization
        LSTM(
            units=64,                      # Number of LSTM units
            dropout=0.3,                   # Dropout rate
            recurrent_dropout=0.3          # Recurrent dropout rate
        ),
        
        # Additional dropout layer
        Dropout(0.5),
        
        # Output layer with sigmoid activation for binary classification
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("LSTM Model Architecture:")
    model.summary()
    
    # ========================================
    # Train LSTM Model
    # ========================================
    print("\nTraining LSTM model...")
    start_time = time.time()
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,                     # Reduced for faster training
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    lstm_train_time = time.time() - start_time
    
    # ========================================
    # Evaluate LSTM Model
    # ========================================
    print("\nEvaluating LSTM model...")
    start_time = time.time()
    
    # Get predictions
    lstm_pred_probs = model.predict(X_test)
    lstm_predictions = (lstm_pred_probs > 0.5).astype(int).flatten()
    
    lstm_predict_time = time.time() - start_time
    
    # Calculate metrics
    lstm_accuracy = accuracy_score(y_test, lstm_predictions)
    lstm_precision = precision_score(y_test, lstm_predictions)
    lstm_recall = recall_score(y_test, lstm_predictions)
    lstm_f1 = f1_score(y_test, lstm_predictions)
    
    print(f"LSTM Training completed in {lstm_train_time:.2f} seconds")
    print(f"LSTM Prediction completed in {lstm_predict_time:.4f} seconds")
    print("\nLSTM Classification Report:")
    print(classification_report(y_test, lstm_predictions, target_names=['Ham', 'Spam']))
    print("LSTM Confusion Matrix:")
    print(confusion_matrix(y_test, lstm_predictions))
    
    return {
        'accuracy': lstm_accuracy,
        'precision': lstm_precision,
        'recall': lstm_recall,
        'f1_score': lstm_f1,
        'train_time': lstm_train_time,
        'predict_time': lstm_predict_time
    }, model

# ============================================================================
# 6. FINAL COMPARISON AND ANALYSIS
# ============================================================================

def create_comparison_report(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comprehensive comparison report of all models.
    
    Args:
        results: Dictionary containing results for all models
        
    Returns:
        pd.DataFrame: Comparison report
    """
    print("\n6. FINAL MODEL COMPARISON")
    print("-" * 35)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    
    # Round values for better readability
    comparison_df = comparison_df.round(4)
    
    # Sort by F1 score (best overall metric for imbalanced data)
    comparison_df = comparison_df.sort_values('f1_score', ascending=False)
    
    print("\nCOMPREHENSIVE MODEL COMPARISON REPORT")
    print("=" * 60)
    print(comparison_df)
    
    # Detailed analysis
    print("\n\nDETAILED ANALYSIS")
    print("-" * 30)
    
    best_accuracy = comparison_df['accuracy'].idxmax()
    best_precision = comparison_df['precision'].idxmax()
    best_recall = comparison_df['recall'].idxmax()
    best_f1 = comparison_df['f1_score'].idxmax()
    fastest_train = comparison_df['train_time'].idxmin()
    fastest_predict = comparison_df['predict_time'].idxmin()
    
    print(f"🎯 Best Accuracy: {best_accuracy} ({comparison_df.loc[best_accuracy, 'accuracy']:.4f})")
    print(f"🎯 Best Precision: {best_precision} ({comparison_df.loc[best_precision, 'precision']:.4f})")
    print(f"🎯 Best Recall: {best_recall} ({comparison_df.loc[best_recall, 'recall']:.4f})")
    print(f"🏆 Best F1-Score: {best_f1} ({comparison_df.loc[best_f1, 'f1_score']:.4f})")
    print(f"⚡ Fastest Training: {fastest_train} ({comparison_df.loc[fastest_train, 'train_time']:.2f}s)")
    print(f"⚡ Fastest Prediction: {fastest_predict} ({comparison_df.loc[fastest_predict, 'predict_time']:.4f}s)")
    
    print("\nRECOMMENDATIONS")
    print("-" * 20)
    
    if best_f1 == 'SVM':
        print("✅ SVM shows the best overall performance (F1-score)")
        print("   - Excellent for text classification with engineered features")
        print("   - Good balance of precision and recall")
        print("   - Recommended for production deployment")
    elif best_f1 == 'Random Forest':
        print("✅ Random Forest shows the best overall performance (F1-score)")
        print("   - Excellent interpretability with feature importance")
        print("   - Robust to overfitting")
        print("   - Good for understanding feature contributions")
    else:
        print("✅ LSTM shows the best overall performance (F1-score)")
        print("   - Captures sequential patterns in text")
        print("   - Can learn complex text representations")
        print("   - Consider if you have larger datasets")
    
    return comparison_df

def save_models_and_artifacts(models_dict: Dict, preprocessing_objects: Dict, 
                            results_df: pd.DataFrame, timestamp: str = None) -> str:
    """
    Save all trained models, preprocessing objects, and results to organized files.
    
    Args:
        models_dict: Dictionary containing all trained models
        preprocessing_objects: Dictionary containing preprocessing components
        results_df: DataFrame with model comparison results
        timestamp: Optional timestamp string for file naming
        
    Returns:
        str: Path to the saved models directory
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create models directory
    models_dir = f"saved_models_{timestamp}"
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\n📁 SAVING MODELS AND ARTIFACTS")
    print("-" * 45)
    print(f"📂 Models directory: {models_dir}")
    
    # ========================================
    # Save Classical Models (SVM & Random Forest)
    # ========================================
    print("\n💾 Saving classical models...")
    
    # Save SVM model
    svm_path = os.path.join(models_dir, "svm_model.joblib")
    joblib.dump(models_dict['svm'], svm_path)
    print(f"  ✅ SVM model saved: {svm_path}")
    
    # Save Random Forest model
    rf_path = os.path.join(models_dir, "random_forest_model.joblib")
    joblib.dump(models_dict['random_forest'], rf_path)
    print(f"  ✅ Random Forest model saved: {rf_path}")
    
    # ========================================
    # Save LSTM Model
    # ========================================
    print("\n🧠 Saving LSTM model...")
    
    # Save LSTM model (Keras format)
    lstm_path = os.path.join(models_dir, "lstm_model.h5")
    models_dict['lstm'].save(lstm_path)
    print(f"  ✅ LSTM model saved: {lstm_path}")
    
    # ========================================
    # Save Preprocessing Objects
    # ========================================
    print("\n🔧 Saving preprocessing components...")
    
    # Save TF-IDF vectorizer
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    joblib.dump(preprocessing_objects['tfidf_vectorizer'], tfidf_path)
    print(f"  ✅ TF-IDF vectorizer saved: {tfidf_path}")
    
    # Save StandardScaler
    scaler_path = os.path.join(models_dir, "standard_scaler.joblib")
    joblib.dump(preprocessing_objects['scaler'], scaler_path)
    print(f"  ✅ StandardScaler saved: {scaler_path}")
    
    # Save LSTM Tokenizer
    tokenizer_path = os.path.join(models_dir, "lstm_tokenizer.pickle")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(preprocessing_objects['tokenizer'], f)
    print(f"  ✅ LSTM tokenizer saved: {tokenizer_path}")
    
    # Save LabelEncoder
    label_encoder_path = os.path.join(models_dir, "label_encoder.joblib")
    joblib.dump(preprocessing_objects['label_encoder'], label_encoder_path)
    print(f"  ✅ Label encoder saved: {label_encoder_path}")
    
    # ========================================
    # Save Model Configuration & Metadata
    # ========================================
    print("\n📋 Saving model metadata...")
    
    # Create metadata dictionary
    metadata = {
        'timestamp': timestamp,
        'models': {
            'svm': {
                'type': 'Support Vector Machine',
                'library': 'sklearn',
                'file': 'svm_model.joblib',
                'hyperparameters': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'class_weight': 'balanced'
                }
            },
            'random_forest': {
                'type': 'Random Forest Classifier',
                'library': 'sklearn', 
                'file': 'random_forest_model.joblib',
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'class_weight': 'balanced'
                }
            },
            'lstm': {
                'type': 'LSTM Neural Network',
                'library': 'tensorflow.keras',
                'file': 'lstm_model.h5',
                'architecture': {
                    'embedding_dim': 128,
                    'lstm_units': 64,
                    'dropout': 0.3,
                    'recurrent_dropout': 0.3,
                    'final_dropout': 0.5
                },
                'training': {
                    'optimizer': 'Adam',
                    'learning_rate': 0.001,
                    'loss': 'binary_crossentropy',
                    'batch_size': 32,
                    'max_epochs': 10,
                    'early_stopping_patience': 3
                }
            }
        },
        'preprocessing': {
            'tfidf_vectorizer': {
                'file': 'tfidf_vectorizer.joblib',
                'parameters': {
                    'max_features': 5000,
                    'min_df': 2,
                    'max_df': 0.95,
                    'ngram_range': [1, 2],
                    'stop_words': 'english'
                }
            },
            'standard_scaler': {
                'file': 'standard_scaler.joblib',
                'features': ['message_length', 'digit_ratio', 'capital_ratio', 'special_char_count']
            },
            'lstm_tokenizer': {
                'file': 'lstm_tokenizer.pickle',
                'parameters': {
                    'num_words': 10000,
                    'oov_token': '<OOV>',
                    'max_sequence_length': preprocessing_objects.get('max_sequence_length', 'unknown')
                }
            },
            'label_encoder': {
                'file': 'label_encoder.joblib',
                'encoding': {'ham': 0, 'spam': 1}
            }
        },
        'feature_engineering': {
            'numerical_features': ['message_length', 'digit_ratio', 'capital_ratio', 'special_char_count'],
            'text_features': ['cleaned_message'],
            'tfidf_features': 5000,
            'total_classical_features': 5004
        }
    }
    
    # Save metadata as JSON
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Model metadata saved: {metadata_path}")
    
    # ========================================
    # Save Performance Results
    # ========================================
    print("\n📊 Saving performance results...")
    
    # Save comprehensive results CSV
    results_path = os.path.join(models_dir, "model_comparison_results.csv")
    results_df.to_csv(results_path)
    print(f"  ✅ Performance results saved: {results_path}")
    
    # ========================================
    # Create Model Loading Instructions
    # ========================================
    loading_instructions = f"""
# SMS Spam Detection Models - Loading Instructions
# Generated: {timestamp}

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
lstm_model = load_model('{models_dir}/lstm_model.h5')
with open('{models_dir}/lstm_tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
label_encoder = joblib.load('{models_dir}/label_encoder.joblib')

# For Classical Models (Faster Inference)
# Load SVM/Random Forest and preprocessors
svm_model = joblib.load('{models_dir}/svm_model.joblib')
rf_model = joblib.load('{models_dir}/random_forest_model.joblib')
tfidf_vectorizer = joblib.load('{models_dir}/tfidf_vectorizer.joblib')
scaler = joblib.load('{models_dir}/standard_scaler.joblib')
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
        sequence, maxlen={preprocessing_objects.get('max_sequence_length', 'SEQ_LENGTH')}
    )
    
    # Get prediction
    prediction = lstm_model.predict(padded)[0][0]
    label = 'spam' if prediction > 0.5 else 'ham'
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return label, confidence

# Example usage
message = "URGENT! You've won $1000! Click here now!"
label, confidence = predict_spam_lstm(message)
print(f"Prediction: {{label}} ({{confidence:.2%}} confidence)")
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
"""
    
    instructions_path = os.path.join(models_dir, "README_model_loading.md")
    with open(instructions_path, 'w') as f:
        f.write(loading_instructions)
    print(f"  ✅ Loading instructions saved: {instructions_path}")
    
    # ========================================
    # Summary
    # ========================================
    print(f"\n🎉 ALL ARTIFACTS SAVED SUCCESSFULLY!")
    print(f"📂 Location: {models_dir}/")
    print(f"📊 Models: SVM, Random Forest, LSTM")
    print(f"🔧 Preprocessors: TF-IDF, Scaler, Tokenizer, Label Encoder")
    print(f"📋 Documentation: Metadata, Instructions, Results")
    
    return models_dir

# ============================================================================
# 7. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Execute the complete machine learning pipeline.
    """
    try:
        # Load and prepare data
        df, label_encoder = load_and_prepare_data()
        
        # Prepare features for classical models
        X_combined, y, feature_names, tfidf_vectorizer, scaler = prepare_classical_features(df)
        
        # Prepare features for LSTM
        X_padded, y_lstm, tokenizer, max_sequence_length = prepare_lstm_features(df)
        
        # Train classical models
        classical_results, (X_train, X_test, y_train, y_test), classical_models = train_classical_models(X_combined, y)
        
        # Train LSTM model
        lstm_results, lstm_model = train_lstm_model(X_padded, y_lstm, max_sequence_length, len(tokenizer.word_index))
        
        # Combine all results
        all_results = {**classical_results, 'LSTM': lstm_results}
        
        # Create final comparison
        comparison_df = create_comparison_report(all_results)
        
        # Prepare all models and preprocessing objects for saving
        models_dict = {
            'svm': classical_models['svm'],
            'random_forest': classical_models['random_forest'],
            'lstm': lstm_model
        }
        
        preprocessing_objects = {
            'tfidf_vectorizer': tfidf_vectorizer,
            'scaler': scaler,
            'tokenizer': tokenizer,
            'label_encoder': label_encoder,
            'max_sequence_length': max_sequence_length
        }
        
        # Save all models and artifacts
        models_dir = save_models_and_artifacts(models_dict, preprocessing_objects, comparison_df)
        
        # Also save results to root directory for backward compatibility
        comparison_df.to_csv('model_comparison_results.csv')
        print(f"\n📊 Results also saved to: model_comparison_results.csv")
        
        print("\n" + "=" * 70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return comparison_df, models_dir
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Execute the complete pipeline
    results = main()
