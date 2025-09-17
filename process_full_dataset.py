"""
Apply Feature Engineering Pipeline to Complete SMS Spam Dataset

This script loads the full spam.csv dataset and applies the complete feature engineering
pipeline to create a comprehensive dataset ready for machine learning.
"""

import pandas as pd
import numpy as np
from feature_engineering_pipeline import apply_feature_engineering, analyze_features
import time

def load_spam_dataset(csv_path: str = 'data/spam.csv') -> pd.DataFrame:
    """
    Load and clean the spam dataset.
    
    Args:
        csv_path (str): Path to the spam.csv file
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with 'label' and 'message' columns
    """
    print("Loading spam dataset...")
    
    # Load the dataset with proper encoding
    df = pd.read_csv(csv_path, encoding='latin-1')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # The dataset has columns: v1 (label), v2 (message), and some unnamed columns
    # Keep only the relevant columns
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'message']
    
    # Remove any null values
    initial_count = len(df)
    df = df.dropna()
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} rows with null values")
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Spam percentage: {(df['label'] == 'spam').mean() * 100:.1f}%")
    
    return df

def process_full_dataset():
    """
    Process the complete spam dataset with feature engineering.
    """
    print("SMS SPAM DETECTION - FULL DATASET PROCESSING")
    print("=" * 60)
    
    # Load the dataset
    df = load_spam_dataset()
    
    # Apply feature engineering pipeline
    print("\n" + "=" * 60)
    print("APPLYING FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Process the full dataset
    df_processed = apply_feature_engineering(df, message_column='message')
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Average processing time per message: {processing_time / len(df) * 1000:.2f} ms")
    
    # Analyze the results
    print("\n" + "=" * 60)
    print("COMPREHENSIVE FEATURE ANALYSIS")
    print("=" * 60)
    
    analyze_features(df_processed, label_column='label')
    
    # Additional analysis for the full dataset
    print("\n" + "=" * 60)
    print("DETAILED SPAM VS HAM COMPARISON")
    print("=" * 60)
    
    # Group by label for detailed comparison
    spam_data = df_processed[df_processed['label'] == 'spam']
    ham_data = df_processed[df_processed['label'] == 'ham']
    
    print(f"\nSpam messages: {len(spam_data)}")
    print(f"Ham messages: {len(ham_data)}")
    
    # Feature comparison
    features = ['message_length', 'digit_ratio', 'capital_ratio', 'special_char_count']
    
    print("\nDetailed Feature Comparison:")
    print("-" * 50)
    
    for feature in features:
        spam_mean = spam_data[feature].mean()
        ham_mean = ham_data[feature].mean()
        spam_std = spam_data[feature].std()
        ham_std = ham_data[feature].std()
        difference = spam_mean - ham_mean
        
        print(f"\n{feature.upper()}:")
        print(f"  Spam:  {spam_mean:.3f} ± {spam_std:.3f}")
        print(f"  Ham:   {ham_mean:.3f} ± {ham_std:.3f}")
        print(f"  Diff:  {difference:.3f} ({'Higher' if difference > 0 else 'Lower'} in spam)")
    
    # Show extreme examples
    print("\n" + "=" * 60)
    print("EXTREME EXAMPLES")
    print("=" * 60)
    
    # Highest digit ratio spam
    highest_digit = df_processed.loc[df_processed['digit_ratio'].idxmax()]
    print(f"\nHighest digit ratio ({highest_digit['digit_ratio']:.3f}):")
    print(f"Label: {highest_digit['label']}")
    print(f"Message: {highest_digit['message'][:100]}...")
    
    # Highest capital ratio spam
    highest_capital = df_processed.loc[df_processed['capital_ratio'].idxmax()]
    print(f"\nHighest capital ratio ({highest_capital['capital_ratio']:.3f}):")
    print(f"Label: {highest_capital['label']}")
    print(f"Message: {highest_capital['message'][:100]}...")
    
    # Longest message
    longest_msg = df_processed.loc[df_processed['message_length'].idxmax()]
    print(f"\nLongest message ({longest_msg['message_length']} chars):")
    print(f"Label: {longest_msg['label']}")
    print(f"Message: {longest_msg['message'][:100]}...")
    
    # Most special characters
    most_special = df_processed.loc[df_processed['special_char_count'].idxmax()]
    print(f"\nMost special characters ({most_special['special_char_count']}):")
    print(f"Label: {most_special['label']}")
    print(f"Message: {most_special['message'][:100]}...")
    
    # Save the processed dataset
    output_path = 'data/spam_with_features.csv'
    df_processed.to_csv(output_path, index=False)
    print(f"\n" + "=" * 60)
    print(f"PROCESSED DATASET SAVED TO: {output_path}")
    print("=" * 60)
    
    print(f"\nFinal dataset shape: {df_processed.shape}")
    print(f"Columns: {df_processed.columns.tolist()}")
    
    # Dataset summary
    print("\nDataset Summary:")
    print("-" * 30)
    print(f"Total messages: {len(df_processed):,}")
    print(f"Spam messages: {len(spam_data):,} ({len(spam_data)/len(df_processed)*100:.1f}%)")
    print(f"Ham messages: {len(ham_data):,} ({len(ham_data)/len(df_processed)*100:.1f}%)")
    print(f"Average message length: {df_processed['message_length'].mean():.1f} characters")
    print(f"Average digit ratio: {df_processed['digit_ratio'].mean():.3f}")
    print(f"Average capital ratio: {df_processed['capital_ratio'].mean():.3f}")
    print(f"Average special characters: {df_processed['special_char_count'].mean():.1f}")
    
    print("\n" + "=" * 60)
    print("READY FOR MACHINE LEARNING!")
    print("=" * 60)
    print("\nThe processed dataset is now ready for:")
    print("1. TF-IDF vectorization on 'cleaned_message' column")
    print("2. Feature scaling for numerical features")
    print("3. Train/test split")
    print("4. Model training (SVM, Naive Bayes, Random Forest, etc.)")
    print("5. Model evaluation and hyperparameter tuning")
    
    return df_processed

if __name__ == "__main__":
    # Process the full dataset
    processed_df = process_full_dataset()
    
    print(f"\nProcessing complete! Dataset ready with {len(processed_df)} messages.")
