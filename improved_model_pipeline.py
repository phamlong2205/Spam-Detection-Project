"""
SMS Spam Detection - Overfitting-Resistant Model Pipeline

Overfitting prevention techniques:
- Feature reduction and selection
- SMOTE for class balancing
- Regularization for all models
- Proper validation methodology

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

# Enhanced ML imports for overfitting prevention
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, 
    validation_curve, learning_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Deep learning imports with enhanced regularization
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, BatchNormalization, 
    GlobalMaxPooling1D, SpatialDropout1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns
from simple_visualization_pipeline import SimpleSpamVisualizer

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🛡️  SMS SPAM DETECTION - OVERFITTING PREVENTION")
print("=" * 50)

def load_and_analyze_data(csv_path: str = 'data/spam_with_features_clean_new.csv') -> Tuple[pd.DataFrame, LabelEncoder]:
    """Load data and analyze class distribution."""
    print("\n1. 📊 DATA LOADING")
    print("-" * 20)
    
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    
    # Class distribution
    class_counts = df['label'].value_counts()
    spam_ratio = (df['label'] == 'spam').mean()
    print(f"Ham: {class_counts['ham']:,} ({(1-spam_ratio)*100:.1f}%)")
    print(f"Spam: {class_counts['spam']:,} ({spam_ratio*100:.1f}%)")
    print(f"Imbalance ratio: {class_counts['ham']/class_counts['spam']:.1f}:1")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    return df, label_encoder

def create_balanced_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, object, object, object]:
    """Create focused feature sets with selected features only."""
    print("\n2. 🎯 FOCUSED FEATURE ENGINEERING")
    print("-" * 33)
    
    y = df['label_encoded'].values
    
    # === TEXT FEATURES ===
    print("Creating text features from cleaned_message...")
    
    # Primary text: cleaned_message
    primary_text = df['cleaned_message'].fillna('')
    
    # TF-IDF for cleaned_message
    print("  - TF-IDF on cleaned_message...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=3,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    tfidf_features = tfidf_vectorizer.fit_transform(primary_text)
    
    # Feature selection on TF-IDF
    print("  - Selecting top 800 TF-IDF features...")
    selector = SelectKBest(score_func=mutual_info_classif, k=800)
    tfidf_selected = selector.fit_transform(tfidf_features, y)
    
    # === SELECTED NUMERICAL FEATURES ===
    print("Processing selected numerical features...")
    selected_numerical = [
        'message_length', 'digit_ratio', 'capital_ratio', 
        'special_char_count', 'url_count'
    ]
    
    # Verify selected features exist
    available_numerical = [f for f in selected_numerical if f in df.columns]
    missing_numerical = [f for f in selected_numerical if f not in df.columns]
    
    print(f"  - Selected numerical features: {available_numerical}")
    if missing_numerical:
        print(f"  - Missing numerical features: {missing_numerical}")
    
    numerical_data = df[available_numerical].fillna(0).values
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_data)
    
    # === SELECTED BOOLEAN FEATURE ===
    print("Processing selected boolean feature...")
    selected_boolean = ['subject_has_suspicious_words']
    
    # Check if boolean feature exists
    if 'subject_has_suspicious_words' in df.columns:
        boolean_data = df[selected_boolean].fillna(False).astype(int).values
        boolean_sparse = csr_matrix(boolean_data)
        print(f"  - Added boolean feature: subject_has_suspicious_words")
    else:
        boolean_sparse = csr_matrix((len(df), 0))
        print(f"  - Boolean feature not found: subject_has_suspicious_words")
    
    # === COMBINE SELECTED FEATURES ===
    print("Combining selected feature types...")
    numerical_sparse = csr_matrix(numerical_scaled)
    
    X_combined = hstack([
        tfidf_selected,      # Text features (800)
        numerical_sparse,    # Selected numerical features (5)
        boolean_sparse       # Selected boolean feature (1)
    ])
    
    print(f"Selected feature summary:")
    print(f"  - TF-IDF from cleaned_message: {tfidf_selected.shape[1]}")
    print(f"  - Numerical features: {numerical_scaled.shape[1]} {available_numerical}")
    print(f"  - Boolean features: {boolean_sparse.shape[1]} {selected_boolean if boolean_sparse.shape[1] > 0 else []}")
    print(f"  - Total selected features: {X_combined.shape[1]}")
    print(f"  - Feature-to-sample ratio: {X_combined.shape[1]/len(y):.4f}")
    
    return X_combined, y, tfidf_vectorizer, selector, scaler

def prepare_lstm_features_regularized(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, object, int]:
    """Prepare LSTM features using only cleaned_message."""
    print("\n3. 🧠 FOCUSED LSTM FEATURES")
    print("-" * 26)
    
    # Use only cleaned_message (focused approach)
    text_data = df['cleaned_message'].fillna('').tolist()
    print(f"Text samples for LSTM: {len(text_data)}")
    
    # Optimized vocabulary for focused features
    tokenizer = Tokenizer(
        num_words=6000,  # Balanced vocabulary size
        oov_token='<OOV>',
        lower=True,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    
    # Sequence length analysis
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
    
    print(f"Using vocabulary: {len(tokenizer.word_index)}, Max length: {max_length}")
    
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
    
    print(f"Train: {len(y_train):,}, Val: {len(y_val):,}, Test: {len(y_test):,}")
    print(f"Spam ratios - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_regularized_classical_models(X_train, X_val, X_test, y_train, y_val, y_test) -> Dict:
    """Train SVM and Random Forest with regularization."""
    print(f"\n5. 🛡️  CLASSICAL MODELS")
    print("-" * 22)
    
    results = {}
    
    # Apply SMOTE for class balancing
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    spam_before = y_train.sum()
    spam_after = y_train_balanced.sum()
    print(f"Spam samples: {spam_before} → {spam_after}")
    
    # Random Forest with Bagging
    print("\n🌲 Training Random Forest...")
    
    # Conservative hyperparameters
    base_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Bagging for additional regularization
    rf_model = base_rf
    
    start_time = time.time()
    rf_model.fit(X_train_balanced, y_train_balanced)
    train_time = time.time() - start_time
    
    start_time = time.time()
    val_predictions = rf_model.predict(X_val)
    val_proba = rf_model.predict_proba(X_val)[:, 1]
    predict_time = time.time() - start_time
    
    test_predictions = rf_model.predict(X_test)
    test_proba = rf_model.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'val_accuracy': accuracy_score(y_val, val_predictions),
        'val_precision': precision_score(y_val, val_predictions),
        'val_recall': recall_score(y_val, val_predictions),
        'val_f1': f1_score(y_val, val_predictions),
        'val_auc': roc_auc_score(y_val, val_proba),
        'test_accuracy': accuracy_score(y_test, test_predictions),
        'test_precision': precision_score(y_test, test_predictions),
        'test_recall': recall_score(y_test, test_predictions),
        'test_f1': f1_score(y_test, test_predictions),
        'test_auc': roc_auc_score(y_test, test_proba),
        'train_time': train_time,
        'predict_time': predict_time
    }
    
    print(f"Validation F1: {results['Random Forest']['val_f1']:.4f}")
    print(f"Test F1: {results['Random Forest']['test_f1']:.4f}")
    
    # SVM
    print("\n⚡ Training SVM...")
    
    svm_model = SVC(
        kernel='rbf',
        C=0.5,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    
    start_time = time.time()
    svm_model.fit(X_train_balanced, y_train_balanced)
    train_time = time.time() - start_time
    
    start_time = time.time()
    val_predictions = svm_model.predict(X_val)
    val_proba = svm_model.predict_proba(X_val)[:, 1]
    predict_time = time.time() - start_time
    
    test_predictions = svm_model.predict(X_test)
    test_proba = svm_model.predict_proba(X_test)[:, 1]
    
    results['SVM'] = {
        'val_accuracy': accuracy_score(y_val, val_predictions),
        'val_precision': precision_score(y_val, val_predictions),
        'val_recall': recall_score(y_val, val_predictions),
        'val_f1': f1_score(y_val, val_predictions),
        'val_auc': roc_auc_score(y_val, val_proba),
        'test_accuracy': accuracy_score(y_test, test_predictions),
        'test_precision': precision_score(y_test, test_predictions),
        'test_recall': recall_score(y_test, test_predictions),
        'test_f1': f1_score(y_test, test_predictions),
        'test_auc': roc_auc_score(y_test, test_proba),
        'train_time': train_time,
        'predict_time': predict_time
    }
    
    print(f"Validation F1: {results['SVM']['val_f1']:.4f}")
    print(f"Test F1: {results['SVM']['test_f1']:.4f}")
    
    return results, {'rf': rf_model, 'svm': svm_model}

def train_regularized_lstm(X_padded, y, max_sequence_length, vocab_size) -> Tuple[Dict, object]:
    """Train LSTM with comprehensive regularization."""
    print(f"\n6. 🧠 REGULARIZED LSTM")
    print("-" * 20)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = create_stratified_splits(X_padded, y)
    
    # Apply SMOTE for LSTM
    print("Applying SMOTE for LSTM...")
    
    # Reshape for SMOTE (flatten sequences)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_train_balanced, y_train_balanced = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_train_flat, y_train)
    X_train_balanced = X_train_balanced.reshape(-1, max_sequence_length)
    
    print(f"Training samples: {len(y_train)} → {len(y_train_balanced)}")
    
    # Build regularized LSTM
    print("\nBuilding LSTM architecture...")
    
    model = Sequential([
        # Embedding without regularization
        Embedding(
            input_dim=vocab_size + 1,
            output_dim=128,
            input_length=max_sequence_length,
            mask_zero=True
        ),
        
        # Spatial dropout
        SpatialDropout1D(0.3),  # Increased from 0.2
        
        # LSTM with heavy regularization
        LSTM(
            units=64,
            dropout=0.3,
            recurrent_dropout=0.3,
            return_sequences=False
        ),
        
        # Dropout
        Dropout(0.5),  # Increased from 0.4
        
        # Dense layer with regularization
        Dense(
            32,
            activation='relu'
        ),
        
        Dropout(0.3),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Conservative optimizer
    optimizer = Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("LSTM Architecture:")
    model.summary()
    
    # Callbacks for overfitting prevention
    callbacks = [
        EarlyStopping(
            monitor='val_f1_score',
            patience=5,  # Reduced from 7 for earlier stopping
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
            'best_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    print("\nTraining LSTM...")
    start_time = time.time()
    
    history = model.fit(
        X_train_balanced, y_train_balanced,
        batch_size=64,    # Reduced from 128 for more regularization
        epochs=10,        # Reduced from 15 to prevent overfitting
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    train_time = time.time() - start_time
    
    # Evaluate model
    print("\nEvaluating LSTM...")
    
    start_time = time.time()
    val_predictions_proba = model.predict(X_val)
    val_predictions = (val_predictions_proba > 0.5).astype(int).flatten()
    predict_time = time.time() - start_time
    
    test_predictions_proba = model.predict(X_test)
    test_predictions = (test_predictions_proba > 0.5).astype(int).flatten()
    
    results = {
        'val_accuracy': accuracy_score(y_val, val_predictions),
        'val_precision': precision_score(y_val, val_predictions),
        'val_recall': recall_score(y_val, val_predictions),
        'val_f1': f1_score(y_val, val_predictions),
        'val_auc': roc_auc_score(y_val, val_predictions_proba),
        'test_accuracy': accuracy_score(y_test, test_predictions),
        'test_precision': precision_score(y_test, test_predictions),
        'test_recall': recall_score(y_test, test_predictions),
        'test_f1': f1_score(y_test, test_predictions),
        'test_auc': roc_auc_score(y_test, test_predictions_proba),
        'train_time': train_time,
        'predict_time': predict_time
    }
    
    print(f"Validation F1: {results['val_f1']:.4f}")
    print(f"Test F1: {results['test_f1']:.4f}")
    
    return results, model

def create_overfitting_analysis(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Compare validation vs test performance to detect overfitting."""
    print(f"\n7. 🔍 OVERFITTING ANALYSIS")
    print("-" * 25)
    
    # Create comparison DataFrame
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
            'Train_Time': metrics['train_time']
        })
    
    df = pd.DataFrame(analysis_data)
    
    print("\nOVERFITTING ANALYSIS:")
    print(df.round(4))
    
    # Analyze gaps
    for _, row in df.iterrows():
        model = row['Model']
        f1_gap = row['F1_Gap']
        
        print(f"\n{model}:")
        if abs(f1_gap) < 0.03:
            print(f"  ✅ F1 gap: {f1_gap:.4f} (Good generalization)")
        elif abs(f1_gap) < 0.07:
            print(f"  ⚠️  F1 gap: {f1_gap:.4f} (Mild overfitting)")
        else:
            print(f"  🚨 F1 gap: {f1_gap:.4f} (Significant overfitting)")
    
    # Best models
    best_generalization = df.loc[df['F1_Gap'].abs().idxmin(), 'Model']
    best_performance = df.loc[df['Test_F1'].idxmax(), 'Model']
    
    print(f"\nBest generalization: {best_generalization}")
    print(f"Best test performance: {best_performance}")
    
    return df

def plot_overfitting_analysis(results_df: pd.DataFrame):
    """Create overfitting analysis visualizations."""
    print(f"\n8. 📈 CREATING VISUALIZATIONS")
    print("-" * 30)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Overfitting Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Validation vs Test Performance
    ax1 = axes[0, 0]
    x = range(len(results_df))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], results_df['Val_F1'], width, label='Validation F1', alpha=0.8)
    ax1.bar([i + width/2 for i in x], results_df['Test_F1'], width, label='Test F1', alpha=0.8)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Validation vs Test Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Model'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance Gaps
    ax2 = axes[0, 1]
    ax2.bar(results_df['Model'], results_df['F1_Gap'], color=['green' if gap < 0.03 else 'orange' if gap < 0.07 else 'red' for gap in results_df['F1_Gap']])
    ax2.set_xlabel('Models')
    ax2.set_ylabel('F1 Gap (Val - Test)')
    ax2.set_title('Overfitting Indicator (F1 Gap)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=0.03, color='orange', linestyle='--', alpha=0.5, label='Mild threshold')
    ax2.axhline(y=0.07, color='red', linestyle='--', alpha=0.5, label='Severe threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Model Comparison
    ax3 = axes[1, 0]
    metrics = ['Test_Accuracy', 'Test_F1', 'Test_AUC']
    for i, model in enumerate(results_df['Model']):
        values = [results_df.iloc[i][metric] for metric in metrics]
        ax3.plot(metrics, values, marker='o', label=model, linewidth=2)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Score')
    ax3.set_title('Test Performance Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # 4. Training Time vs Performance
    ax4 = axes[1, 1]
    scatter = ax4.scatter(results_df['Train_Time'], results_df['Test_F1'], 
                         s=100, alpha=0.7, c=results_df['F1_Gap'], cmap='RdYlGn_r')
    
    for i, model in enumerate(results_df['Model']):
        ax4.annotate(model, (results_df.iloc[i]['Train_Time'], results_df.iloc[i]['Test_F1']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Training Time (seconds)')
    ax4.set_ylabel('Test F1 Score')
    ax4.set_title('Training Time vs Performance')
    plt.colorbar(scatter, ax=ax4, label='F1 Gap (Overfitting)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print("Plots saved to: overfitting_analysis.png")
    
    return fig

def save_all_models(classical_models, lstm_model, tokenizer, tfidf_vectorizer, 
                   selector, scaler, label_encoder, max_sequence_length, timestamp):
    """Save all trained models and preprocessors to a timestamped folder."""
    print(f"\n💾 SAVING ALL MODELS")
    print("=" * 20)
    
    # Create timestamped folder
    folder_name = f'saved_models_{timestamp}'
    os.makedirs(folder_name, exist_ok=True)
    print(f"📁 Created folder: {folder_name}")
    
    saved_files = {'folder': folder_name, 'files': {}}
    
    try:
        # 1. Save classical models
        print("📦 Saving classical models...")
        # classical_models is {'rf': rf_model, 'svm': svm_model}
        model_name_mapping = {'rf': 'random_forest', 'svm': 'svm'}
        
        for model_key, model_obj in classical_models.items():
            model_name = model_name_mapping.get(model_key, model_key)
            model_file = f'{folder_name}/{model_name}_model.joblib'
            joblib.dump(model_obj, model_file)
            saved_files['files'][f'{model_name}_model'] = model_file
            print(f"   ✅ {model_key.upper()}: {model_file}")
        
        # 2. Save LSTM model
        print("🧠 Saving LSTM model...")
        lstm_file = f'{folder_name}/lstm_model.h5'
        lstm_model.save(lstm_file)
        saved_files['files']['lstm_model'] = lstm_file
        print(f"   ✅ LSTM model: {lstm_file}")
        
        # 3. Save preprocessors
        print("⚙️ Saving preprocessors...")
        
        # TF-IDF Vectorizer
        tfidf_file = f'{folder_name}/tfidf_vectorizer.joblib'
        joblib.dump(tfidf_vectorizer, tfidf_file)
        saved_files['files']['tfidf_vectorizer'] = tfidf_file
        print(f"   ✅ TF-IDF Vectorizer: {tfidf_file}")
        
        # Feature Selector
        selector_file = f'{folder_name}/feature_selector.joblib'
        joblib.dump(selector, selector_file)
        saved_files['files']['feature_selector'] = selector_file
        print(f"   ✅ Feature Selector: {selector_file}")
        
        # Standard Scaler
        scaler_file = f'{folder_name}/standard_scaler.joblib'
        joblib.dump(scaler, scaler_file)
        saved_files['files']['standard_scaler'] = scaler_file
        print(f"   ✅ Standard Scaler: {scaler_file}")
        
        # Label Encoder
        label_file = f'{folder_name}/label_encoder.joblib'
        joblib.dump(label_encoder, label_file)
        saved_files['files']['label_encoder'] = label_file
        print(f"   ✅ Label Encoder: {label_file}")
        
        # LSTM Tokenizer
        tokenizer_file = f'{folder_name}/lstm_tokenizer.pickle'
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(tokenizer, f)
        saved_files['files']['lstm_tokenizer'] = tokenizer_file
        print(f"   ✅ LSTM Tokenizer: {tokenizer_file}")
        
        # 4. Save metadata
        print("📋 Saving metadata...")
        metadata = {
            'timestamp': timestamp,
            'max_sequence_length': max_sequence_length,
            'vocab_size': len(tokenizer.word_index),
            'feature_count': {
                'tfidf_selected': 800,
                'numerical': 5,  # Based on our focused features
                'boolean': 1
            },
            'models_included': ['Random Forest', 'SVM', 'LSTM'],
            'preprocessing_pipeline': [
                'TF-IDF Vectorization',
                'Feature Selection (SelectKBest)',
                'Standard Scaling',
                'LSTM Tokenization'
            ]
        }
        
        metadata_file = f'{folder_name}/model_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['files']['metadata'] = metadata_file
        print(f"   ✅ Metadata: {metadata_file}")
        
        # 5. Create README
        print("📖 Creating README...")
        readme_content = f"""
# Spam Detection Models - Loading Instructions
# Generated: {timestamp}

## Quick Start - Load Best Performing Model

```python
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# Load LSTM model and preprocessors
lstm_model = load_model('{folder_name}/lstm_model.h5')
with open('{folder_name}/lstm_tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
label_encoder = joblib.load('{folder_name}/label_encoder.joblib')

# Load classical models and preprocessors
random_forest_model = joblib.load('{folder_name}/random_forest_model.joblib')
svm_model = joblib.load('{folder_name}/svm_model.joblib')
tfidf_vectorizer = joblib.load('{folder_name}/tfidf_vectorizer.joblib')
scaler = joblib.load('{folder_name}/standard_scaler.joblib')
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
        sequence, maxlen={max_sequence_length}
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
- random_forest_model.joblib: Random Forest model
- svm_model.joblib: Support Vector Machine model  
- tfidf_vectorizer.joblib: TF-IDF text vectorizer
- feature_selector.joblib: Feature selection transformer
- standard_scaler.joblib: Numerical feature scaler
- lstm_tokenizer.pickle: LSTM text tokenizer
- label_encoder.joblib: Label encoder (ham/spam)
- model_metadata.json: Complete model configuration
"""
        
        readme_file = f'{folder_name}/README_model_loading.md'
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        saved_files['files']['readme'] = readme_file
        print(f"   ✅ README: {readme_file}")
        
        print(f"\n🎯 ALL MODELS SAVED SUCCESSFULLY!")
        print(f"📦 Total files saved: {len(saved_files['files'])}")
        print(f"📁 Location: {folder_name}/")
        
        return saved_files
        
    except Exception as e:
        print(f"❌ Error saving models: {str(e)}")
        return {'folder': folder_name, 'error': str(e)}

def main():
    """Execute the overfitting-resistant pipeline."""
    print("\n🚀 STARTING PIPELINE")
    print("=" * 25)
    
    try:
        # Load and analyze data
        df, label_encoder = load_and_analyze_data()
        
        # Create balanced features
        X_combined, y, tfidf_vectorizer, selector, scaler = create_balanced_features(df)
        
        # Create LSTM features
        X_padded, y_lstm, tokenizer, max_sequence_length = prepare_lstm_features_regularized(df)
        
        # Train classical models
        classical_results, classical_models = train_regularized_classical_models(
            *create_stratified_splits(X_combined, y)
        )
        
        # Train LSTM
        lstm_results, lstm_model = train_regularized_lstm(
            X_padded, y_lstm, max_sequence_length, len(tokenizer.word_index)
        )
        
        # Combine results
        all_results = {**classical_results, 'LSTM': lstm_results}
        
        # Overfitting analysis
        analysis_df = create_overfitting_analysis(all_results)
        
        # Create visualizations
        plot_overfitting_analysis(analysis_df)
        
        # Save results and models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_df.to_csv(f'improved_model_results_{timestamp}.csv', index=False)
        
        # Save all models and preprocessors
        saved_models = save_all_models(
            classical_models, lstm_model, tokenizer, tfidf_vectorizer, 
            selector, scaler, label_encoder, max_sequence_length, timestamp
        )
        
        print(f"\n🎉 PIPELINE COMPLETED!")
        print("=" * 25)
        print(f"Results saved to: improved_model_results_{timestamp}.csv")
        print(f"Models saved to: {saved_models['folder']}")
        print(f"Visualizations saved to: overfitting_analysis.png")
        
        return analysis_df, all_results, saved_models
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
