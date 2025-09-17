"""
SMS Spam Detection Dataset Processing Demo

This script demonstrates how to use the SMS preprocessing pipeline 
with the spam.csv dataset to prepare data for machine learning.
"""

import pandas as pd
from sms_preprocessor import SMSPreprocessor, preprocess_sms_text
import time

def load_and_preprocess_spam_data(csv_path: str, sample_size: int = None):
    """
    Load and preprocess the spam dataset.
    
    Args:
        csv_path (str): Path to the spam.csv file
        sample_size (int, optional): Number of samples to process (for testing)
        
    Returns:
        pd.DataFrame: DataFrame with original and preprocessed text
    """
    # Load the dataset
    df = pd.read_csv(csv_path, encoding='latin-1')
    
    # The dataset has columns: v1 (label), v2 (message), and some unnamed columns
    # Let's clean it up
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'message']
    
    # Remove any null values
    df = df.dropna()
    
    # Take a sample if specified
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print("\n" + "="*50)
    
    # Initialize preprocessor
    preprocessor = SMSPreprocessor(use_lemmatization=True)
    
    # Preprocess messages
    print("Preprocessing messages...")
    start_time = time.time()
    
    df['processed_message'] = df['message'].apply(preprocessor.preprocess_sms)
    
    end_time = time.time()
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds")
    print(f"Average processing time per message: {(end_time - start_time) / len(df) * 1000:.2f} ms")
    
    return df

def show_preprocessing_examples(df: pd.DataFrame, n_examples: int = 5):
    """
    Display examples of original and preprocessed messages.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        n_examples (int): Number of examples to show
    """
    print(f"\n{n_examples} Random Preprocessing Examples:")
    print("="*80)
    
    # Show examples from both spam and ham
    spam_examples = df[df['label'] == 'spam'].sample(n=min(n_examples//2 + 1, sum(df['label'] == 'spam')))
    ham_examples = df[df['label'] == 'ham'].sample(n=min(n_examples//2 + 1, sum(df['label'] == 'ham')))
    
    examples = pd.concat([spam_examples, ham_examples]).head(n_examples)
    
    for idx, row in examples.iterrows():
        print(f"\nExample {idx + 1} [{row['label'].upper()}]:")
        print(f"Original: {row['message'][:100]}{'...' if len(row['message']) > 100 else ''}")
        print(f"Processed: {row['processed_message']}")
        print("-" * 80)

def analyze_preprocessing_results(df: pd.DataFrame):
    """
    Analyze the results of preprocessing.
    
    Args:
        df (pd.DataFrame): Processed dataframe
    """
    print("\nPreprocessing Analysis:")
    print("="*50)
    
    # Calculate statistics
    original_lengths = df['message'].str.len()
    processed_lengths = df['processed_message'].str.len()
    
    print(f"Original message length - Mean: {original_lengths.mean():.1f}, Median: {original_lengths.median():.1f}")
    print(f"Processed message length - Mean: {processed_lengths.mean():.1f}, Median: {processed_lengths.median():.1f}")
    print(f"Average length reduction: {(1 - processed_lengths.mean() / original_lengths.mean()) * 100:.1f}%")
    
    # Word count analysis
    original_word_counts = df['message'].str.split().str.len()
    processed_word_counts = df['processed_message'].str.split().str.len()
    
    print(f"\nOriginal word count - Mean: {original_word_counts.mean():.1f}, Median: {original_word_counts.median():.1f}")
    print(f"Processed word count - Mean: {processed_word_counts.mean():.1f}, Median: {processed_word_counts.median():.1f}")
    print(f"Average word reduction: {(1 - processed_word_counts.mean() / original_word_counts.mean()) * 100:.1f}%")
    
    # Empty messages after preprocessing
    empty_processed = (df['processed_message'].str.strip() == '').sum()
    print(f"\nMessages that became empty after preprocessing: {empty_processed} ({empty_processed/len(df)*100:.2f}%)")

def save_processed_data(df: pd.DataFrame, output_path: str):
    """
    Save the processed dataset.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        output_path (str): Path to save the processed data
    """
    df.to_csv(output_path, index=False)
    print(f"\nProcessed dataset saved to: {output_path}")

if __name__ == "__main__":
    # Process the dataset
    print("SMS Spam Detection - Dataset Preprocessing Demo")
    print("="*60)
    
    # Load and preprocess data (using a sample for demo)
    df = load_and_preprocess_spam_data('spam.csv', sample_size=100)
    
    # Show examples
    show_preprocessing_examples(df, n_examples=6)
    
    # Analyze results
    analyze_preprocessing_results(df)
    
    # Save processed data
    save_processed_data(df, 'processed_spam_sample.csv')
    
    print("\n" + "="*60)
    print("Demo completed! The preprocessing pipeline is ready for use with machine learning models.")
    print("\nNext steps:")
    print("1. Use TF-IDF vectorization on the 'processed_message' column")
    print("2. Split data into training and testing sets")
    print("3. Train your spam detection model (SVM, Naive Bayes, etc.)")
    print("4. Evaluate model performance")