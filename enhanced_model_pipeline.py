"""
Enhanced SMS/Email Spam Detection Pipeline - Optimized for New Rich Dataset
==========================================================================

Features optimized for spam_with_features_clean_new.csv with 18 features:
- Text features: cleaned_message, subject
- Numerical: message_length, digit_ratio, capital_ratio, special_char_count, 
            average_word_length, url_count, max_consecutive_special_chars
- Boolean: subject_is_all_caps, subject_has_suspicious_words, reply_to_mismatch, has_attachment
- Categorical: message_type (email/sms)

Date: 2025-09-26
"""

import pandas as pd
import numpy as np
import time
import joblib
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Enhanced ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTE

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, BatchNormalization, SpatialDropout1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import F1Score

import matplotlib.pyplot as plt
import seaborn as sns
from simple_visualization_pipeline import SimpleSpamVisualizer

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🚀 ENHANCED SPAM DETECTION - RICH FEATURE PIPELINE")
print("=" * 55)

def load_and_analyze_enhanced_data(csv_path: str = 'data/spam_with_features_clean_new.csv') -> Tuple[pd.DataFrame, LabelEncoder]:
    """Load enhanced dataset and analyze class distribution."""
    print("\n1. 📊 DATA LOADING & ANALYSIS")
    print("-" * 30)
    
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Features available: {df.columns.tolist()}")
    
    # Class distribution
    class_counts = df['label'].value_counts()
    spam_ratio = (df['label'] == 'spam').mean()
    print(f"\nClass Distribution:")
    print(f"Ham: {class_counts['ham']:,} ({(1-spam_ratio)*100:.1f}%)")
    print(f"Spam: {class_counts['spam']:,} ({spam_ratio*100:.1f}%)")
    print(f"Imbalance ratio: {class_counts['ham']/class_counts['spam']:.1f}:1")
    
    # Feature completeness check
    print(f"\nFeature Completeness:")
    missing_features = df.isnull().sum()
    for feature in missing_features[missing_features > 0].index:
        missing_pct = (missing_features[feature] / len(df)) * 100
        print(f"  {feature}: {missing_features[feature]:,} missing ({missing_pct:.1f}%)")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    return df, label_encoder

def create_enhanced_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Create comprehensive feature set from enhanced dataset."""
    print("\n2. 🛠️  ENHANCED FEATURE ENGINEERING")
    print("-" * 35)
    
    # Target variable
    y = df['label_encoded'].values
    
    # === TEXT FEATURES ===
    print("Creating text features...")
    
    # Primary text: cleaned_message
    primary_text = df['cleaned_message'].fillna('')
    
    # Secondary text: subject (for emails)
    subject_text = df['subject'].fillna('')
    
    # Combined text (cleaned_message + subject)
    combined_text = primary_text + ' ' + subject_text
    
    # TF-IDF for primary text (cleaned_message)
    print("  - TF-IDF on cleaned_message...")
    tfidf_primary = TfidfVectorizer(
        max_features=1500,
        min_df=3,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    tfidf_primary_features = tfidf_primary.fit_transform(primary_text)
    
    # TF-IDF for subject lines
    print("  - TF-IDF on subject lines...")
    tfidf_subject = TfidfVectorizer(
        max_features=300,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    tfidf_subject_features = tfidf_subject.fit_transform(subject_text)
    
    # === NUMERICAL FEATURES ===
    print("Processing numerical features...")
    numerical_features = [
        'message_length', 'digit_ratio', 'capital_ratio', 'special_char_count',
        'average_word_length', 'url_count', 'max_consecutive_special_chars'
    ]
    
    numerical_data = df[numerical_features].fillna(0).values
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_data)
    
    # === BOOLEAN FEATURES ===
    print("Processing boolean features...")
    boolean_features = [
        'subject_is_all_caps', 'subject_has_suspicious_words', 
        'reply_to_mismatch', 'has_attachment'
    ]
    
    boolean_data = df[boolean_features].fillna(False).astype(int).values
    
    # === CATEGORICAL FEATURES ===
    print("Processing categorical features...")
    
    # Message type (email/sms) - one-hot encode
    message_type_encoder = LabelBinarizer()
    message_type_encoded = message_type_encoder.fit_transform(df['message_type'].fillna('unknown'))
    
    # === FEATURE COMBINATION ===
    print("Combining all features...")
    
    # Convert numerical and boolean to sparse
    numerical_sparse = csr_matrix(numerical_scaled)
    boolean_sparse = csr_matrix(boolean_data)
    categorical_sparse = csr_matrix(message_type_encoded)
    
    # Combine all features
    X_combined = hstack([
        tfidf_primary_features,    # Text features (1500)
        tfidf_subject_features,    # Subject features (300) 
        numerical_sparse,          # Numerical features (7)
        boolean_sparse,            # Boolean features (4)
        categorical_sparse         # Categorical features (2-3)
    ])
    
    print(f"Final feature matrix shape: {X_combined.shape}")
    print(f"Feature breakdown:")
    print(f"  - TF-IDF (primary): {tfidf_primary_features.shape[1]}")
    print(f"  - TF-IDF (subject): {tfidf_subject_features.shape[1]}")
    print(f"  - Numerical: {numerical_scaled.shape[1]}")
    print(f"  - Boolean: {boolean_data.shape[1]}")
    print(f"  - Categorical: {message_type_encoded.shape[1]}")
    print(f"  - Total: {X_combined.shape[1]}")
    print(f"  - Feature-to-sample ratio: {X_combined.shape[1]/len(y):.4f}")
    
    # Store preprocessing objects
    preprocessing_objects = {
        'tfidf_primary': tfidf_primary,
        'tfidf_subject': tfidf_subject,
        'scaler': scaler,
        'message_type_encoder': message_type_encoder,
        'feature_names': {
            'numerical': numerical_features,
            'boolean': boolean_features
        }
    }
    
    return X_combined, y, preprocessing_objects

def prepare_enhanced_lstm_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, object, int]:
    """Prepare LSTM features with enhanced text combination."""
    print("\n3. 🧠 ENHANCED LSTM FEATURES")
    print("-" * 28)
    
    # Combine cleaned_message and subject for richer text representation
    primary_text = df['cleaned_message'].fillna('')
    subject_text = df['subject'].fillna('')
    
    # Create enhanced text by combining message and subject
    enhanced_text = []
    for i in range(len(df)):
        combined = primary_text.iloc[i]
        if subject_text.iloc[i].strip():  # Add subject if not empty
            combined = subject_text.iloc[i] + ' ' + combined
        enhanced_text.append(combined)
    
    print(f"Enhanced text samples created: {len(enhanced_text)}")
    
    # Tokenization with expanded vocabulary for richer dataset
    tokenizer = Tokenizer(
        num_words=8000,  # Increased for richer dataset
        oov_token='<OOV>',
        lower=True,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    
    tokenizer.fit_on_texts(enhanced_text)
    sequences = tokenizer.texts_to_sequences(enhanced_text)
    
    # Determine optimal sequence length
    sequence_lengths = [len(seq) for seq in sequences]
    avg_length = np.mean(sequence_lengths)
    percentile_90 = int(np.percentile(sequence_lengths, 90))
    percentile_95 = int(np.percentile(sequence_lengths, 95))
    
    print(f"Sequence statistics:")
    print(f"  - Average length: {avg_length:.1f}")
    print(f"  - 90th percentile: {percentile_90}")
    print(f"  - 95th percentile: {percentile_95}")
    
    # Use 90th percentile for balance of information and efficiency
    max_length = percentile_90
    
    print(f"Using max sequence length: {max_length}")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    
    X_padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    y = df['label_encoded'].values
    
    return X_padded, y, tokenizer, max_length

def create_stratified_splits(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Create stratified train/validation/test splits."""
    print(f"\n4. 🎯 DATA SPLITTING")
    print("-" * 18)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"Data splits:")
    print(f"  - Train: {len(y_train):,} samples ({y_train.mean():.3f} spam ratio)")
    print(f"  - Val: {len(y_val):,} samples ({y_val.mean():.3f} spam ratio)")
    print(f"  - Test: {len(y_test):,} samples ({y_test.mean():.3f} spam ratio)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_enhanced_classical_models(X_train, X_val, X_test, y_train, y_val, y_test) -> Dict:
    """Train enhanced classical models on rich feature set."""
    print(f"\n5. 🤖 ENHANCED CLASSICAL MODELS")
    print("-" * 30)
    
    results = {}
    
    # Apply SMOTE for class balancing
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    spam_before = y_train.sum()
    spam_after = y_train_balanced.sum()
    print(f"  Spam samples: {spam_before:,} → {spam_after:,}")
    print(f"  Total samples: {len(y_train):,} → {len(y_train_balanced):,}")
    
    # === RANDOM FOREST ===
    print("\n🌲 Training Enhanced Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=200,           # More trees for rich feature set
        max_depth=15,              # Deeper for complex patterns
        min_samples_split=5,       # Less restrictive
        min_samples_leaf=2,        # Less restrictive  
        max_features='sqrt',       # Good for high-dimensional data
        class_weight='balanced',   # Handle any remaining imbalance
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    rf_model.fit(X_train_balanced, y_train_balanced)
    rf_train_time = time.time() - start_time
    
    start_time = time.time()
    rf_val_pred = rf_model.predict(X_val)
    rf_val_proba = rf_model.predict_proba(X_val)[:, 1]
    rf_predict_time = time.time() - start_time
    
    rf_test_pred = rf_model.predict(X_test)
    rf_test_proba = rf_model.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'val_accuracy': accuracy_score(y_val, rf_val_pred),
        'val_precision': precision_score(y_val, rf_val_pred),
        'val_recall': recall_score(y_val, rf_val_pred),
        'val_f1': f1_score(y_val, rf_val_pred),
        'val_auc': roc_auc_score(y_val, rf_val_proba),
        'test_accuracy': accuracy_score(y_test, rf_test_pred),
        'test_precision': precision_score(y_test, rf_test_pred),
        'test_recall': recall_score(y_test, rf_test_pred),
        'test_f1': f1_score(y_test, rf_test_pred),
        'test_auc': roc_auc_score(y_test, rf_test_proba),
        'train_time': rf_train_time,
        'predict_time': rf_predict_time
    }
    
    print(f"  Validation F1: {results['Random Forest']['val_f1']:.4f}")
    print(f"  Test F1: {results['Random Forest']['test_f1']:.4f}")
    print(f"  Training time: {rf_train_time:.2f}s")
    
    # === SVM ===
    print("\n⚡ Training Enhanced SVM...")
    
    svm_model = SVC(
        kernel='rbf',
        C=1.0,                    # Balanced regularization
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    
    start_time = time.time()
    svm_model.fit(X_train_balanced, y_train_balanced)
    svm_train_time = time.time() - start_time
    
    start_time = time.time()
    svm_val_pred = svm_model.predict(X_val)
    svm_val_proba = svm_model.predict_proba(X_val)[:, 1]
    svm_predict_time = time.time() - start_time
    
    svm_test_pred = svm_model.predict(X_test)
    svm_test_proba = svm_model.predict_proba(X_test)[:, 1]
    
    results['SVM'] = {
        'val_accuracy': accuracy_score(y_val, svm_val_pred),
        'val_precision': precision_score(y_val, svm_val_pred),
        'val_recall': recall_score(y_val, svm_val_pred),
        'val_f1': f1_score(y_val, svm_val_pred),
        'val_auc': roc_auc_score(y_val, svm_val_proba),
        'test_accuracy': accuracy_score(y_test, svm_test_pred),
        'test_precision': precision_score(y_test, svm_test_pred),
        'test_recall': recall_score(y_test, svm_test_pred),
        'test_f1': f1_score(y_test, svm_test_pred),
        'test_auc': roc_auc_score(y_test, svm_test_proba),
        'train_time': svm_train_time,
        'predict_time': svm_predict_time
    }
    
    print(f"  Validation F1: {results['SVM']['val_f1']:.4f}")
    print(f"  Test F1: {results['SVM']['test_f1']:.4f}")
    print(f"  Training time: {svm_train_time:.2f}s")
    
    return results, {'rf': rf_model, 'svm': svm_model}

def train_enhanced_lstm(X_padded, y, max_sequence_length, vocab_size) -> Tuple[Dict, object]:
    """Train enhanced LSTM model."""
    print(f"\n6. 🧠 ENHANCED LSTM MODEL")
    print("-" * 25)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = create_stratified_splits(X_padded, y)
    
    # Apply SMOTE for LSTM
    print("Applying SMOTE for LSTM...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_train_balanced, y_train_balanced = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train_flat, y_train)
    X_train_balanced = X_train_balanced.reshape(-1, max_sequence_length)
    
    print(f"  Training samples: {len(y_train):,} → {len(y_train_balanced):,}")
    
    # Build enhanced LSTM architecture
    print("\nBuilding enhanced LSTM architecture...")
    
    model = Sequential([
        # Embedding without regularization
        Embedding(
            input_dim=vocab_size + 1,
            output_dim=128,             # Increased for richer features
            input_length=max_sequence_length,
            mask_zero=True
        ),
        
        # Spatial dropout
        SpatialDropout1D(0.2),
        
        # LSTM layer
        LSTM(
            units=64,                   # Increased capacity
            dropout=0.3,
            recurrent_dropout=0.3,
            return_sequences=False
        ),
        
        # Dropout
        Dropout(0.4),
        
        # Dense layer
        Dense(
            32,
            activation='relu'
        ),
        
        Dropout(0.3),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Optimizer
    optimizer = Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )
    
    # Compile with F1 metric
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', F1Score()]
    )
    
    print("Enhanced LSTM Architecture:")
    model.summary()
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_f1_score',
            patience=7,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        
        ModelCheckpoint(
            'enhanced_best_lstm_model.h5',
            monitor='val_f1_score',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]
    
    # Training
    print("\nTraining enhanced LSTM...")
    start_time = time.time()
    
    history = model.fit(
        X_train_balanced, y_train_balanced,
        batch_size=128,
        epochs=15,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    lstm_train_time = time.time() - start_time
    
    # Evaluation
    print("\nEvaluating enhanced LSTM...")
    
    start_time = time.time()
    val_pred_proba = model.predict(X_val)
    val_predictions = (val_pred_proba > 0.5).astype(int).flatten()
    lstm_predict_time = time.time() - start_time
    
    test_pred_proba = model.predict(X_test)
    test_predictions = (test_pred_proba > 0.5).astype(int).flatten()
    
    results = {
        'val_accuracy': accuracy_score(y_val, val_predictions),
        'val_precision': precision_score(y_val, val_predictions),
        'val_recall': recall_score(y_val, val_predictions),
        'val_f1': f1_score(y_val, val_predictions),
        'val_auc': roc_auc_score(y_val, val_pred_proba),
        'test_accuracy': accuracy_score(y_test, test_predictions),
        'test_precision': precision_score(y_test, test_predictions),
        'test_recall': recall_score(y_test, test_predictions),
        'test_f1': f1_score(y_test, test_predictions),
        'test_auc': roc_auc_score(y_test, test_pred_proba),
        'train_time': lstm_train_time,
        'predict_time': lstm_predict_time
    }
    
    print(f"  Validation F1: {results['val_f1']:.4f}")
    print(f"  Test F1: {results['test_f1']:.4f}")
    print(f"  Training time: {lstm_train_time:.2f}s")
    
    return results, model

def create_performance_analysis(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create comprehensive performance analysis."""
    print(f"\n7. 📊 PERFORMANCE ANALYSIS")
    print("-" * 27)
    
    # Create analysis DataFrame
    analysis_data = []
    
    for model_name, metrics in results.items():
        analysis_data.append({
            'Model': model_name,
            'Val_Accuracy': metrics['val_accuracy'],
            'Test_Accuracy': metrics['test_accuracy'],
            'Accuracy_Gap': metrics['val_accuracy'] - metrics['test_accuracy'],
            'Val_F1': metrics['val_f1'],
            'Test_F1': metrics['test_f1'],
            'F1_Gap': metrics['val_f1'] - metrics['test_f1'],
            'Val_AUC': metrics['val_auc'],
            'Test_AUC': metrics['test_auc'],
            'AUC_Gap': metrics['val_auc'] - metrics['test_auc'],
            'Train_Time': metrics['train_time'],
            'Predict_Time': metrics['predict_time']
        })
    
    df = pd.DataFrame(analysis_data)
    
    print("\nPERFORMANCE COMPARISON:")
    print("=" * 40)
    print(df.round(4))
    
    # Analysis insights
    best_f1 = df.loc[df['Test_F1'].idxmax(), 'Model']
    fastest = df.loc[df['Train_Time'].idxmin(), 'Model']
    best_generalization = df.loc[df['F1_Gap'].abs().idxmin(), 'Model']
    
    print(f"\n🏆 PERFORMANCE INSIGHTS:")
    print(f"  Best Test F1: {best_f1} ({df['Test_F1'].max():.4f})")
    print(f"  Fastest Training: {fastest} ({df['Train_Time'].min():.2f}s)")
    print(f"  Best Generalization: {best_generalization}")
    
    # Overfitting analysis
    print(f"\n🔍 GENERALIZATION ANALYSIS:")
    for _, row in df.iterrows():
        f1_gap = row['F1_Gap']
        model = row['Model']
        if abs(f1_gap) < 0.02:
            status = "✅ Excellent"
        elif abs(f1_gap) < 0.05:
            status = "⚠️  Good"
        else:
            status = "🚨 Concerning"
        print(f"  {model}: {status} (Gap: {f1_gap:.4f})")
    
    return df

def main():
    """Execute the enhanced spam detection pipeline."""
    print("\n🚀 STARTING ENHANCED PIPELINE")
    print("=" * 35)
    
    try:
        # Load enhanced data
        df, label_encoder = load_and_analyze_enhanced_data()
        
        # Create enhanced features
        X_combined, y, preprocessing_objects = create_enhanced_features(df)
        
        # Create enhanced LSTM features
        X_padded, y_lstm, tokenizer, max_sequence_length = prepare_enhanced_lstm_features(df)
        
        # Train classical models
        classical_results, classical_models = train_enhanced_classical_models(
            *create_stratified_splits(X_combined, y)
        )
        
        # Train LSTM
        lstm_results, lstm_model = train_enhanced_lstm(
            X_padded, y_lstm, max_sequence_length, len(tokenizer.word_index)
        )
        
        # Combine results
        all_results = {**classical_results, 'Enhanced LSTM': lstm_results}
        
        # Performance analysis
        analysis_df = create_performance_analysis(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'enhanced_model_results_{timestamp}.csv'
        analysis_df.to_csv(results_file, index=False)
        
        # Generate visualizations
        print(f"\n8. 🎨 GENERATING VISUALIZATIONS")
        print("-" * 30)
        visualizer = SimpleSpamVisualizer('data/spam_with_features_clean_new.csv')
        visualizer.create_dataset_analysis()
        visualizer.create_model_comparison(results_file)
        visualizer.create_training_progress()
        
        print(f"\n🎉 ENHANCED PIPELINE COMPLETED!")
        print("=" * 40)
        print(f"📊 Results saved: {results_file}")
        print(f"🎨 Visualizations generated:")
        print(f"   • simple_dataset_analysis.png")
        print(f"   • simple_model_comparison.png")
        print(f"   • simple_training_progress.png")
        
        print(f"\n💡 RECOMMENDATIONS:")
        best_model = analysis_df.loc[analysis_df['Test_F1'].idxmax(), 'Model']
        print(f"   • Deploy: {best_model} for production")
        print(f"   • Monitor: F1 gaps for generalization")
        print(f"   • Consider: Ensemble of top 2 models")
        
        return analysis_df, all_results
        
    except Exception as e:
        print(f"\n❌ Error in enhanced pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
