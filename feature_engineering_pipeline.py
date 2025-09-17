"""
SMS Spam Detection Feature Engineering Pipeline

This script implements a comprehensive feature engineering pipeline for SMS spam detection.
It creates both cleaned text for TF-IDF vectorization and extracts metadata features
that are strong indicators of spam messages.

"""

import pandas as pd
import numpy as np
import string
import re
from typing import Union, List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    print("NLTK data download complete!")


def preprocess_text(text: str) -> str:
    """
    Preprocess raw text for TF-IDF vectorization by applying standard NLP cleaning steps.
    
    This function performs the core text preprocessing steps that prepare text for 
    machine learning models:
    
    1. Convert to lowercase - normalizes text case
    2. Remove punctuation - reduces noise in feature space
    3. Tokenize - splits text into individual words
    4. Remove English stop words - eliminates common words with little discriminative power
    5. Apply Porter stemming - reduces words to their root forms for better generalization
    
    Args:
        text (str): Raw SMS message text
        
    Returns:
        str: Cleaned and preprocessed text ready for TF-IDF vectorization
        
    Example:
        >>> preprocess_text("Hello! You've WON a FREE prize! Call now!!!")
        'hello won free prize call'
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Step 1: Convert to lowercase
    text = text.lower()
    
    # Step 2: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Step 3: Tokenize into words
    tokens = word_tokenize(text)
    
    # Step 4: Remove English stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Step 5: Apply Porter stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Join back into a single string for TF-IDF
    return ' '.join(stemmed_tokens)


def calculate_message_length(text: str) -> int:
    """
    Calculate the total character count of the message.
    
    Message length is a key spam indicator because:
    - Spam messages are often longer to include multiple offers, urgency tactics, and contact info
    - Legitimate personal messages tend to be shorter and more conversational
    - Marketing spam often includes detailed terms, conditions, and legal disclaimers
    - Phishing attempts may be lengthy to appear more legitimate
    
    Args:
        text (str): Original raw message text
        
    Returns:
        int: Total number of characters in the message
        
    Example:
        >>> calculate_message_length("Free entry! Call 123-456-7890 now!")
        34
    """
    if not isinstance(text, str):
        return 0
    return len(text)


def calculate_digit_ratio(text: str) -> float:
    """
    Calculate the proportion of the message that consists of numeric digits.
    
    Digit ratio is a strong spam indicator because:
    - Spam often contains phone numbers for contact (high digit content)
    - Promotional messages include prices, discount percentages, and offer codes
    - Contest/lottery spam includes claim codes and phone numbers
    - Legitimate personal messages typically have minimal numeric content
    - Values above 0.1 (10% digits) are often spam indicators
    
    Args:
        text (str): Original raw message text
        
    Returns:
        float: Ratio of digits to total characters (0.0 to 1.0)
        
    Example:
        >>> calculate_digit_ratio("Call 123-456-7890 for 50% off!")
        0.32  # 10 digits out of 31 total characters
    """
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    
    digit_count = sum(1 for char in text if char.isdigit())
    return digit_count / len(text)


def calculate_capital_ratio(text: str) -> float:
    """
    Calculate the proportion of the message that is uppercase letters.
    
    Capital ratio is a spam indicator because:
    - Spam uses excessive capitalization to create urgency ("FREE", "WINNER", "URGENT")
    - Legitimate messages typically follow normal capitalization rules
    - ALL CAPS text is a common spam tactic to grab attention
    - Values above 0.3 (30% capitals) often indicate spam
    - Personal messages rarely have high capital ratios except for emphasis
    
    Args:
        text (str): Original raw message text
        
    Returns:
        float: Ratio of uppercase letters to total letters (0.0 to 1.0)
        
    Example:
        >>> calculate_capital_ratio("FREE WINNER! You've won $1000!")
        0.43  # 10 uppercase letters out of 23 total letters
    """
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    
    # Count only alphabetic characters for the denominator
    letters = [char for char in text if char.isalpha()]
    if len(letters) == 0:
        return 0.0
    
    uppercase_count = sum(1 for char in letters if char.isupper())
    return uppercase_count / len(letters)


def calculate_special_char_count(text: str) -> int:
    """
    Count special characters commonly used in spam messages.
    
    Special character count is a spam indicator because:
    - '$' symbols indicate monetary offers, prices, and financial scams
    - '!' creates urgency and excitement ("Act now!", "Limited time!")
    - '%' symbols often appear in discount offers ("50% off", "90% savings")
    - '@' symbols may indicate email addresses or social media handles
    - Multiple special characters suggest promotional/commercial content
    - Personal messages typically use minimal special characters
    
    The function counts: $ ! % @
    
    Args:
        text (str): Original raw message text
        
    Returns:
        int: Total count of special characters ($, !, %, @)
        
    Example:
        >>> calculate_special_char_count("50% off! $100 value! Email us @ contact@spam.com")
        5  # 1 '%' + 2 '!' + 1 '$' + 1 '@'
    """
    if not isinstance(text, str):
        return 0
    
    # Define special characters commonly found in spam
    special_chars = ['$', '!', '%', '@']
    
    # Count occurrences of each special character
    total_count = sum(text.count(char) for char in special_chars)
    
    return total_count


def apply_feature_engineering(df: pd.DataFrame, 
                            message_column: str = 'message',
                            inplace: bool = False) -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline to a DataFrame.
    
    This function transforms a DataFrame with raw SMS messages into a feature-rich
    dataset ready for machine learning. It creates both cleaned text for TF-IDF
    and metadata features that are strong spam indicators.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing SMS messages
        message_column (str): Name of the column containing raw message text
        inplace (bool): If True, modify the DataFrame in place; if False, return a copy
        
    Returns:
        pd.DataFrame: DataFrame with new feature columns added:
            - cleaned_message: Preprocessed text for TF-IDF
            - message_length: Character count
            - digit_ratio: Proportion of digits
            - capital_ratio: Proportion of uppercase letters  
            - special_char_count: Count of special characters
            
    Raises:
        ValueError: If message_column doesn't exist in the DataFrame
        
    Example:
        >>> df = pd.DataFrame({'message': ['Free entry! Call 123-456-7890']})
        >>> enriched_df = apply_feature_engineering(df)
        >>> print(enriched_df.columns.tolist())
        ['message', 'cleaned_message', 'message_length', 'digit_ratio', 
         'capital_ratio', 'special_char_count']
    """
    if message_column not in df.columns:
        raise ValueError(f"Column '{message_column}' not found in DataFrame")
    
    # Create a copy if not modifying in place
    if not inplace:
        df = df.copy()
    
    print("Applying feature engineering pipeline...")
    print(f"Processing {len(df)} messages...")
    
    # Apply text preprocessing for TF-IDF
    print("1/5 Creating cleaned text for TF-IDF...")
    df['cleaned_message'] = df[message_column].apply(preprocess_text)
    
    # Apply metadata feature extraction on original text
    print("2/5 Calculating message lengths...")
    df['message_length'] = df[message_column].apply(calculate_message_length)
    
    print("3/5 Calculating digit ratios...")
    df['digit_ratio'] = df[message_column].apply(calculate_digit_ratio)
    
    print("4/5 Calculating capital ratios...")
    df['capital_ratio'] = df[message_column].apply(calculate_capital_ratio)
    
    print("5/5 Counting special characters...")
    df['special_char_count'] = df[message_column].apply(calculate_special_char_count)
    
    print("Feature engineering complete!")
    print(f"Added 5 new columns: cleaned_message, message_length, digit_ratio, capital_ratio, special_char_count")
    
    return df


def analyze_features(df: pd.DataFrame, label_column: Optional[str] = None) -> None:
    """
    Analyze the engineered features and their distributions.
    
    Args:
        df (pd.DataFrame): DataFrame with engineered features
        label_column (str, optional): Name of the label column for spam/ham analysis
    """
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    # Check if required columns exist
    required_cols = ['cleaned_message', 'message_length', 'digit_ratio', 
                    'capital_ratio', 'special_char_count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        return
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Messages processed: {len(df)}")
    
    # Basic statistics for numerical features
    print("\nNumerical Feature Statistics:")
    print("-" * 40)
    numerical_features = ['message_length', 'digit_ratio', 'capital_ratio', 'special_char_count']
    stats = df[numerical_features].describe()
    print(stats.round(3))
    
    # Feature distributions by class if label column provided
    if label_column and label_column in df.columns:
        print(f"\nFeature Statistics by {label_column.upper()}:")
        print("-" * 50)
        
        for feature in numerical_features:
            print(f"\n{feature.upper()}:")
            group_stats = df.groupby(label_column)[feature].agg(['mean', 'median', 'std']).round(3)
            print(group_stats)
    
    # Text preprocessing statistics
    print("\nText Preprocessing Results:")
    print("-" * 40)
    original_lengths = df['message'].str.len() if 'message' in df.columns else None
    cleaned_lengths = df['cleaned_message'].str.len()
    
    if original_lengths is not None:
        print(f"Average original length: {original_lengths.mean():.1f} characters")
        print(f"Average cleaned length: {cleaned_lengths.mean():.1f} characters")
        print(f"Average length reduction: {(1 - cleaned_lengths.mean() / original_lengths.mean()) * 100:.1f}%")
    
    empty_cleaned = (df['cleaned_message'].str.strip() == '').sum()
    print(f"Messages that became empty after cleaning: {empty_cleaned} ({empty_cleaned/len(df)*100:.2f}%)")


def demonstrate_pipeline():
    """
    Demonstrate the feature engineering pipeline with example data.
    """
    print("SMS SPAM DETECTION - FEATURE ENGINEERING PIPELINE DEMO")
    print("="*70)
    
    # Create sample data with typical spam and ham messages
    sample_data = {
        'message': [
            # Spam examples
            "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
            "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
            "Urgent! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
            "50% DISCOUNT! Buy now and SAVE £££! Text SAVE to 85233 or call 09061743806. £1.50 per msg. Customer Service: 08717168528",
            
            # Ham examples  
            "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
            "Ok lar... Joking wif u oni...",
            "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.",
            "Hey can you pick me up at the airport tomorrow at 3pm? Flight AA123 from Chicago. Thanks!",
            "Meeting moved to 2pm in conference room B. Please bring the quarterly reports."
        ],
        'label': ['spam', 'spam', 'spam', 'spam', 'ham', 'ham', 'ham', 'ham', 'ham']
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    print(f"Created sample dataset with {len(df)} messages")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Apply feature engineering
    print("\n" + "-"*70)
    df_enriched = apply_feature_engineering(df, message_column='message')
    
    # Display examples
    print("\n" + "="*70)
    print("FEATURE ENGINEERING EXAMPLES")
    print("="*70)
    
    # Show a few examples with all features
    for idx in [0, 4, 6]:  # Show spam, ham examples
        row = df_enriched.iloc[idx]
        print(f"\nExample {idx + 1} [{row['label'].upper()}]:")
        print(f"Original: {row['message'][:80]}{'...' if len(row['message']) > 80 else ''}")
        print(f"Cleaned:  {row['cleaned_message']}")
        print(f"Length: {row['message_length']} | Digits: {row['digit_ratio']:.3f} | Capitals: {row['capital_ratio']:.3f} | Special chars: {row['special_char_count']}")
        print("-" * 70)
    
    # Analyze features
    analyze_features(df_enriched, label_column='label')
    
    # Show feature correlation with spam
    print("\n" + "="*60)
    print("SPAM CORRELATION ANALYSIS")
    print("="*60)
    
    # Convert label to binary for correlation
    df_enriched['is_spam'] = (df_enriched['label'] == 'spam').astype(int)
    
    features = ['message_length', 'digit_ratio', 'capital_ratio', 'special_char_count']
    correlations = df_enriched[features + ['is_spam']].corr()['is_spam'].drop('is_spam')
    
    print("Correlation with spam (1=spam, 0=ham):")
    print("-" * 40)
    for feature, corr in correlations.items():
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "positive" if corr > 0 else "negative"
        print(f"{feature:18}: {corr:6.3f} ({strength} {direction})")
    
    print("\n" + "="*70)
    print("PIPELINE READY FOR MACHINE LEARNING!")
    print("="*70)
    print("\nNext steps:")
    print("1. Use 'cleaned_message' column for TF-IDF vectorization")
    print("2. Use metadata features (length, ratios, counts) as additional input features")
    print("3. Combine TF-IDF features with metadata features for model training")
    print("4. Train classifier (SVM, Naive Bayes, Random Forest, etc.)")
    print("5. Evaluate performance using cross-validation")
    
    return df_enriched


# Main execution
if __name__ == "__main__":
    # Run the demonstration
    demo_df = demonstrate_pipeline()
    
    print(f"\n\nDemo completed! Sample DataFrame shape: {demo_df.shape}")
    print("Columns:", demo_df.columns.tolist())
    
    # Example of how to use with your own data
    print("\n" + "="*70)
    print("HOW TO USE WITH YOUR DATA")
    print("="*70)
    print("""
# Load your data
df = pd.read_csv('your_sms_data.csv')

# Apply feature engineering
df_features = apply_feature_engineering(df, message_column='message')

# Analyze the results
analyze_features(df_features, label_column='label')

# The resulting DataFrame will have these columns:
# - original columns (e.g., 'message', 'label')
# - 'cleaned_message': for TF-IDF vectorization
# - 'message_length': character count
# - 'digit_ratio': proportion of digits (0.0 to 1.0)
# - 'capital_ratio': proportion of uppercase letters (0.0 to 1.0) 
# - 'special_char_count': count of $, !, %, @ characters

# Ready for machine learning!
    """)
