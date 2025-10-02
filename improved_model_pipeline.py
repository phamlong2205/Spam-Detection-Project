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

def load_and_analyze_data(csv_path: str = 'data/spam_with_features_clean.csv') -> Tuple[pd.DataFrame, LabelEncoder]:
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
    """Create feature sets with dimensionality reduction."""
    print("\n2. 🛡️  FEATURE ENGINEERING")
    print("-" * 25)
    
    text_data = df['cleaned_message'].fillna('')
    numerical_features = ['message_length', 'digit_ratio', 'capital_ratio', 'special_char_count']
    numerical_data = df[numerical_features].values
    y = df['label_encoded'].values
    
    # Reduced TF-IDF features
    print("Creating TF-IDF features (1000 max)...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=3,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    
    tfidf_features = tfidf_vectorizer.fit_transform(text_data)
    
    # Feature selection
    print("Selecting top 800 features...")
    selector = SelectKBest(score_func=mutual_info_classif, k=800)
    tfidf_selected = selector.fit_transform(tfidf_features, y)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_data)
    
    # Combine features
    numerical_sparse = csr_matrix(numerical_scaled)
    X_combined = hstack([tfidf_selected, numerical_sparse])
    
    print(f"Final features: {X_combined.shape[1]} (ratio: {X_combined.shape[1]/len(y):.3f})")
    
    return X_combined, y, tfidf_vectorizer, selector, scaler

def prepare_lstm_features_regularized(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, object, int]:
    """Prepare LSTM features with reduced vocabulary."""
    print("\n3. 🧠 LSTM FEATURES")
    print("-" * 18)
    
    text_data = df['cleaned_message'].fillna('').tolist()
    
    # Reduced vocabulary
    tokenizer = Tokenizer(
        num_words=5000,
        oov_token='<OOV>',
        lower=True,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    
    # Conservative sequence length (90th percentile)
    sequence_lengths = [len(seq) for seq in sequences]
    max_length = int(np.percentile(sequence_lengths, 90))
    
    print(f"Vocabulary: {len(tokenizer.word_index)}, Max length: {max_length}")
    
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
        SpatialDropout1D(0.2),
        
        # LSTM with heavy regularization
        LSTM(
            units=64,
            dropout=0.3,
            recurrent_dropout=0.3,
            return_sequences=False
        ),
        
        # Dropout
        Dropout(0.4),
        
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
        batch_size=128,
        epochs=15,
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
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_df.to_csv(f'improved_model_results_{timestamp}.csv', index=False)
        
        print(f"\n🎉 PIPELINE COMPLETED!")
        print("=" * 25)
        print(f"Results saved to: improved_model_results_{timestamp}.csv")
        print(f"Visualizations saved to: overfitting_analysis.png")
        
        return analysis_df, all_results
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
