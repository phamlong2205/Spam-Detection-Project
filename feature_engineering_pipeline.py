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


def calculate_average_word_length(text: str) -> float:
    """
    Calculate the average length of words in the message.
    
    Average word length is a spam indicator because:
    - Spam often uses short, punchy words for impact ("WIN", "FREE", "NOW")
    - Legitimate messages tend to have more varied word lengths
    - Very short average word length may indicate excessive use of abbreviations
    - Very long average word length may indicate technical jargon or scam attempts
    
    Args:
        text (str): Original raw message text
        
    Returns:
        float: Average word length, 0.0 if no words found
        
    Example:
        >>> calculate_average_word_length("Free winner call now")
        4.25  # (4+6+4+3)/4 = 4.25
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    if not words:
        return 0.0
    
    total_length = sum(len(word) for word in words)
    return total_length / len(words)


def calculate_url_count(text: str) -> int:
    """
    Count the number of URLs found in the message.
    
    URL count is a strong spam indicator because:
    - Spam often contains malicious links to phishing sites
    - Multiple URLs suggest commercial/promotional content
    - Legitimate personal messages rarely contain URLs
    - Email spam frequently uses URL shorteners to hide destinations
    - SMS spam uses URLs to bypass character limits
    
    Detects common URL patterns including:
    - http:// and https:// URLs
    - www. domains
    - Common TLDs (.com, .org, .net, etc.)
    - URL shorteners (bit.ly, tinyurl.com, etc.)
    
    Args:
        text (str): Original raw message text
        
    Returns:
        int: Total number of URLs found
        
    Example:
        >>> calculate_url_count("Visit http://example.com or www.spam.org for deals!")
        2
    """
    if not isinstance(text, str):
        return 0
    
    # Comprehensive URL pattern matching
    url_patterns = [
        r'https?://[^\s]+',  # http:// or https:// URLs
        r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # www. domains
        r'\b[a-zA-Z0-9.-]+\.(com|org|net|edu|gov|mil|int|co\.uk|de|fr|jp|cn|au|ca|ru|br|in|nl|pl|es|it|se|no|dk|fi|be|ch|at|cz|hu|ro|bg|hr|si|sk|lt|lv|ee|mt|cy|lu|ie|pt|gr|tr|il|ae|sa|eg|ma|ng|za|ke|gh|tz|ug|zm|zw|mw|sz|ls|bw|na|ao|mz|mg|mu|sc|cv|st|gq|ga|cg|cd|cf|cm|td|ne|ml|bf|ci|sn|gm|gw|sl|lr|gn|mr|eh)\b',  # Common TLDs
        r'\b(bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|short\.link|tiny\.cc)/\S+',  # URL shorteners
    ]
    
    url_count = 0
    for pattern in url_patterns:
        matches = re.findall(pattern, text.lower())
        url_count += len(matches)
    
    return url_count


def calculate_max_consecutive_special_chars(text: str) -> int:
    """
    Find the longest run of consecutive special characters.
    
    Consecutive special characters are a spam indicator because:
    - Spam uses multiple exclamation marks for urgency ("!!!", "!!!!")
    - Dollar signs are repeated for emphasis ("$$$", "$$$$")
    - Creates visual impact and attention-grabbing effect
    - Legitimate messages use special characters more sparingly
    - Common spam patterns: "!!!", "$$$", "***", "%%%"
    
    Special characters counted: ! @ # $ % ^ & * ( ) - _ = + [ ] { } | \ : ; " ' < > , . ? /
    
    Args:
        text (str): Original raw message text
        
    Returns:
        int: Length of longest consecutive special character sequence
        
    Example:
        >>> calculate_max_consecutive_special_chars("WIN $$$!!! Call NOW****")
        4  # "****" is the longest sequence
    """
    if not isinstance(text, str):
        return 0
    
    # Define special characters (excluding spaces and alphanumeric)
    special_chars = set('!@#$%^&*()-_=+[]{}|\\:;"\'<>,.?/')
    
    max_consecutive = 0
    current_consecutive = 0
    
    for char in text:
        if char in special_chars:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive


def check_subject_is_all_caps(subject: Optional[str]) -> bool:
    """
    Check if the email subject line is entirely in uppercase.
    
    All-caps subjects are a spam indicator because:
    - Spam uses ALL CAPS to grab attention and create urgency
    - Legitimate emails rarely use all-caps subjects
    - Professional communication avoids all-caps (considered shouting)
    - Typical spam subjects: "FREE MONEY", "URGENT ACTION REQUIRED"
    
    Args:
        subject (str, optional): Email subject line, None for SMS
        
    Returns:
        bool: True if subject exists and is all uppercase, False otherwise
        
    Example:
        >>> check_subject_is_all_caps("FREE WINNER ANNOUNCEMENT")
        True
        >>> check_subject_is_all_caps("Meeting tomorrow at 2pm")
        False
        >>> check_subject_is_all_caps(None)  # SMS case
        False
    """
    if not subject or not isinstance(subject, str):
        return False
    
    # Remove whitespace and check if any letters exist
    subject_clean = subject.strip()
    if not subject_clean:
        return False
    
    # Check if there are any alphabetic characters
    letters = [char for char in subject_clean if char.isalpha()]
    if not letters:
        return False
    
    # Check if all letters are uppercase
    return all(char.isupper() for char in letters)


def check_subject_has_suspicious_words(subject: Optional[str]) -> bool:
    """
    Check if the email subject contains suspicious spam-related words.
    
    Suspicious words in subjects are spam indicators because:
    - Common spam keywords indicate promotional/scam content
    - These words are frequently used in phishing attempts
    - Legitimate emails rarely use these attention-grabbing terms
    - Pattern matching helps identify marketing/spam campaigns
    
    Suspicious words include: free, winner, urgent, limited, act, now, congratulations,
    prize, reward, offer, deal, discount, save, money, cash, credit, loan, debt,
    click, call, text, claim, guaranteed, risk-free, no obligation
    
    Args:
        subject (str, optional): Email subject line, None for SMS
        
    Returns:
        bool: True if subject contains suspicious words, False otherwise
        
    Example:
        >>> check_subject_has_suspicious_words("FREE WINNER - Claim your prize!")
        True
        >>> check_subject_has_suspicious_words("Meeting agenda for tomorrow")
        False
        >>> check_subject_has_suspicious_words(None)  # SMS case
        False
    """
    if not subject or not isinstance(subject, str):
        return False
    
    # Comprehensive list of suspicious spam keywords
    suspicious_words = {
        'free', 'winner', 'urgent', 'limited', 'act', 'now', 'congratulations',
        'prize', 'reward', 'offer', 'deal', 'discount', 'save', 'money', 'cash',
        'credit', 'loan', 'debt', 'click', 'call', 'text', 'claim', 'guaranteed',
        'risk-free', 'no obligation', 'exclusive', 'special', 'bonus', 'jackpot',
        'lottery', 'sweepstake', 'contest', 'promotion', 'marketing', 'advertisement',
        'buy', 'purchase', 'order', 'shop', 'sale', 'clearance', 'final',
        'expire', 'expires', 'expiry', 'deadline', 'hurry', 'rush', 'fast',
        'instant', 'immediate', 'asap', 'today', 'tonight', 'weekend'
    }
    
    # Convert subject to lowercase and split into words
    subject_words = re.findall(r'\b\w+\b', subject.lower())
    
    # Check if any suspicious words are present
    return any(word in suspicious_words for word in subject_words)


def check_has_attachment(attachment_count: Optional[int] = None, attachment_list: Optional[List[str]] = None) -> bool:
    """
    Check if the email has attachments.
    
    Attachments are a spam indicator because:
    - Malware is often distributed via email attachments
    - Phishing attempts use attachments to bypass email filters
    - Legitimate personal emails less frequently have attachments
    - Business spam often includes fake invoices, documents, or executable files
    - Suspicious file types: .exe, .zip, .rar, .scr, .bat, .com, .pif
    
    Args:
        attachment_count (int, optional): Number of attachments
        attachment_list (List[str], optional): List of attachment filenames
        
    Returns:
        bool: True if email has attachments, False otherwise
        
    Example:
        >>> check_has_attachment(attachment_count=2)
        True
        >>> check_has_attachment(attachment_list=['document.pdf', 'image.jpg'])
        True
        >>> check_has_attachment()  # SMS case
        False
    """
    # Check by count
    if attachment_count is not None:
        return isinstance(attachment_count, int) and attachment_count > 0
    
    # Check by list
    if attachment_list is not None:
        return isinstance(attachment_list, list) and len(attachment_list) > 0
    
    # Default case (SMS or no attachment info)
    return False


def extract_comprehensive_features(message: str,
                                 message_type: str = 'sms',
                                 subject: Optional[str] = None,
                                 from_address: Optional[str] = None,
                                 attachment_count: Optional[int] = None,
                                 attachment_list: Optional[List[str]] = None) -> dict:
    """
    Extract comprehensive features from email or SMS messages.
    
    This function calculates both existing and new features for spam detection,
    handling both email and SMS message types gracefully. Email-specific features
    default to False for SMS messages.
    
    Args:
        message (str): Main message body text
        message_type (str): Type of message ('email' or 'sms')
        subject (str, optional): Email subject line (email only)
        from_address (str, optional): Email From header (email only)
        attachment_count (int, optional): Number of attachments (email only)
        attachment_list (List[str], optional): List of attachment names (email only)
        
    Returns:
        dict: Dictionary containing all calculated features:
            - cleaned_message (str): Preprocessed text for TF-IDF
            - message_length (int): Character count
            - digit_ratio (float): Proportion of digits
            - capital_ratio (float): Proportion of uppercase letters
            - special_char_count (int): Count of special characters
            - average_word_length (float): Average length of words
            - url_count (int): Number of URLs found
            - max_consecutive_special_chars (int): Longest special character sequence
            - subject_is_all_caps (bool): True if subject is all caps
            - subject_has_suspicious_words (bool): True if subject has spam words
            - has_attachment (bool): True if email has attachments
    """
    features = {}
    
    # Existing features (work for both email and SMS)
    features['cleaned_message'] = preprocess_text(message)
    features['message_length'] = calculate_message_length(message)
    features['digit_ratio'] = calculate_digit_ratio(message)
    features['capital_ratio'] = calculate_capital_ratio(message)
    features['special_char_count'] = calculate_special_char_count(message)
    
    # New universal features (work for both email and SMS)
    features['average_word_length'] = calculate_average_word_length(message)
    features['url_count'] = calculate_url_count(message)
    features['max_consecutive_special_chars'] = calculate_max_consecutive_special_chars(message)
    
    # Email-specific features (default to False for SMS)
    if message_type.lower() == 'email':
        features['subject_is_all_caps'] = check_subject_is_all_caps(subject)
        features['subject_has_suspicious_words'] = check_subject_has_suspicious_words(subject)
        features['has_attachment'] = check_has_attachment(attachment_count, attachment_list)
    else:
        # SMS defaults
        features['subject_is_all_caps'] = False
        features['subject_has_suspicious_words'] = False
        features['has_attachment'] = False
    
    return features


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
            - average_word_length: Average length of words
            - url_count: Number of URLs found
            - max_consecutive_special_chars: Longest special character sequence
            
    Raises:
        ValueError: If message_column doesn't exist in the DataFrame
        
    Example:
        >>> df = pd.DataFrame({'message': ['Free entry! Call 123-456-7890']})
        >>> enriched_df = apply_feature_engineering(df)
        >>> print(enriched_df.columns.tolist())
        ['message', 'cleaned_message', 'message_length', 'digit_ratio', 
         'capital_ratio', 'special_char_count', 'average_word_length',
         'url_count', 'max_consecutive_special_chars']
    """
    if message_column not in df.columns:
        raise ValueError(f"Column '{message_column}' not found in DataFrame")
    
    # Create a copy if not modifying in place
    if not inplace:
        df = df.copy()
    
    print("Applying feature engineering pipeline...")
    print(f"Processing {len(df)} messages...")
    
    # Apply text preprocessing for TF-IDF
    print("1/8 Creating cleaned text for TF-IDF...")
    df['cleaned_message'] = df[message_column].apply(preprocess_text)
    
    # Apply metadata feature extraction on original text
    print("2/8 Calculating message lengths...")
    df['message_length'] = df[message_column].apply(calculate_message_length)
    
    print("3/8 Calculating digit ratios...")
    df['digit_ratio'] = df[message_column].apply(calculate_digit_ratio)
    
    print("4/8 Calculating capital ratios...")
    df['capital_ratio'] = df[message_column].apply(calculate_capital_ratio)
    
    print("5/8 Counting special characters...")
    df['special_char_count'] = df[message_column].apply(calculate_special_char_count)
    
    print("6/8 Calculating average word lengths...")
    df['average_word_length'] = df[message_column].apply(calculate_average_word_length)
    
    print("7/8 Counting URLs...")
    df['url_count'] = df[message_column].apply(calculate_url_count)
    
    print("8/8 Finding max consecutive special characters...")
    df['max_consecutive_special_chars'] = df[message_column].apply(calculate_max_consecutive_special_chars)
    
    print("Feature engineering complete!")
    print(f"Added 8 new columns: cleaned_message, message_length, digit_ratio, capital_ratio, special_char_count, average_word_length, url_count, max_consecutive_special_chars")
    
    return df


def apply_comprehensive_feature_engineering(df: pd.DataFrame,
                                          message_column: str = 'message',
                                          message_type_column: Optional[str] = None,
                                          subject_column: Optional[str] = None,
                                          from_column: Optional[str] = None,
                                          attachment_count_column: Optional[str] = None,
                                          attachment_list_column: Optional[str] = None,
                                          inplace: bool = False) -> pd.DataFrame:
    """
    Apply comprehensive feature engineering pipeline to a DataFrame with email/SMS data.
    
    This function works with both email and SMS data, automatically handling
    email-specific features while providing sensible defaults for SMS messages.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        message_column (str): Column containing message body
        message_type_column (str, optional): Column indicating 'email' or 'sms'
        subject_column (str, optional): Column containing email subjects
        from_column (str, optional): Column containing From addresses
        attachment_count_column (str, optional): Column containing attachment counts
        attachment_list_column (str, optional): Column containing attachment lists
        inplace (bool): Whether to modify DataFrame in place
        
    Returns:
        pd.DataFrame: DataFrame with all feature columns added
    """
    if message_column not in df.columns:
        raise ValueError(f"Column '{message_column}' not found in DataFrame")
    
    if not inplace:
        df = df.copy()
    
    print("Applying comprehensive feature engineering pipeline...")
    print(f"Processing {len(df)} messages...")
    
    # Extract features for each row
    features_list = []
    
    for idx, row in df.iterrows():
        # Get message type
        msg_type = 'sms'  # default
        if message_type_column and message_type_column in df.columns:
            msg_type = row[message_type_column] if pd.notna(row[message_type_column]) else 'sms'
        
        # Get optional email fields
        subject = row[subject_column] if subject_column and subject_column in df.columns else None
        from_addr = row[from_column] if from_column and from_column in df.columns else None
        attach_count = row[attachment_count_column] if attachment_count_column and attachment_count_column in df.columns else None
        attach_list = row[attachment_list_column] if attachment_list_column and attachment_list_column in df.columns else None
        
        # Extract features
        features = extract_comprehensive_features(
            message=row[message_column],
            message_type=msg_type,
            subject=subject,
            from_address=from_addr,
            attachment_count=attach_count,
            attachment_list=attach_list
        )
        
        features_list.append(features)
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(df)} messages...")
    
    # Add features to DataFrame
    for feature_name in features_list[0].keys():
        df[feature_name] = [features[feature_name] for features in features_list]
    
    print("Comprehensive feature engineering complete!")
    feature_names = list(features_list[0].keys())
    print(f"Added {len(feature_names)} feature columns: {', '.join(feature_names)}")
    
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
    
    # Add new features if they exist
    new_features = ['average_word_length', 'url_count', 'max_consecutive_special_chars']
    for feature in new_features:
        if feature in df.columns:
            numerical_features.append(feature)
    
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
    
    # Boolean feature analysis if they exist
    boolean_features = ['subject_is_all_caps', 'subject_has_suspicious_words', 'has_attachment']
    existing_boolean_features = [f for f in boolean_features if f in df.columns]
    
    if existing_boolean_features:
        print(f"\nBoolean Feature Analysis:")
        print("-" * 40)
        for feature in existing_boolean_features:
            print(f"{feature}: {df[feature].sum()} True values ({df[feature].mean()*100:.1f}%)")
            
            if label_column and label_column in df.columns:
                true_rate_by_class = df.groupby(label_column)[feature].mean()
                print(f"  True rate by {label_column}: {true_rate_by_class.to_dict()}")
    
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
            "WIN $$$!!! Visit http://bit.ly/win-now for your FREE prize!!! Act NOW!!!",
            
            # Ham examples  
            "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
            "Ok lar... Joking wif u oni...",
            "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.",
            "Hey can you pick me up at the airport tomorrow at 3pm? Flight AA123 from Chicago. Thanks!",
            "Meeting moved to 2pm in conference room B. Please bring the quarterly reports."
        ],
        'label': ['spam', 'spam', 'spam', 'spam', 'spam', 'ham', 'ham', 'ham', 'ham', 'ham']
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
    for idx in [0, 4, 5, 8]:  # Show spam and ham examples
        row = df_enriched.iloc[idx]
        print(f"\nExample {idx + 1} [{row['label'].upper()}]:")
        print(f"Original: {row['message'][:80]}{'...' if len(row['message']) > 80 else ''}")
        print(f"Cleaned:  {row['cleaned_message']}")
        print(f"Features:")
        print(f"  Length: {row['message_length']} chars")
        print(f"  Digit ratio: {row['digit_ratio']:.3f}")
        print(f"  Capital ratio: {row['capital_ratio']:.3f}")
        print(f"  Special chars: {row['special_char_count']}")
        print(f"  Avg word length: {row['average_word_length']:.2f}")
        print(f"  URL count: {row['url_count']}")
        print(f"  Max consecutive special: {row['max_consecutive_special_chars']}")
        print("-" * 70)
    
    # Analyze features
    analyze_features(df_enriched, label_column='label')
    
    # Show feature correlation with spam
    print("\n" + "="*60)
    print("SPAM CORRELATION ANALYSIS")
    print("="*60)
    
    # Convert label to binary for correlation
    df_enriched['is_spam'] = (df_enriched['label'] == 'spam').astype(int)
    
    features = ['message_length', 'digit_ratio', 'capital_ratio', 'special_char_count', 
               'average_word_length', 'url_count', 'max_consecutive_special_chars']
    correlations = df_enriched[features + ['is_spam']].corr()['is_spam'].drop('is_spam')
    
    print("Correlation with spam (1=spam, 0=ham):")
    print("-" * 40)
    for feature, corr in correlations.items():
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "positive" if corr > 0 else "negative"
        print(f"{feature:25}: {corr:6.3f} ({strength} {direction})")
    
    print("\n" + "="*70)
    print("COMPREHENSIVE FEATURE DEMO")
    print("="*70)
    
    # Demonstrate comprehensive features with email example
    email_example = {
        'message': "CONGRATULATIONS! You've won $10,000!!! Visit www.scam.com to claim!!!",
        'message_type': 'email',
        'subject': 'FREE MONEY WINNER!!!',
        'from_address': 'support@bank.com',
        'attachment_count': 1
    }
    
    email_features = extract_comprehensive_features(**email_example)
    
    print("Email example features:")
    for feature, value in email_features.items():
        print(f"  {feature}: {value}")
    
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
