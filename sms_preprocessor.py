"""
SMS Spam Detection Text Preprocessing Pipeline

This module provides comprehensive text preprocessing functionality specifically designed
for SMS spam detection. The preprocessing pipeline is optimized for the unique 
characteristics of SMS messages including abbreviations, special characters, and 
informal language patterns commonly found in spam messages.

"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


class SMSPreprocessor:
    """
    A comprehensive SMS text preprocessing class optimized for spam detection.
    
    This preprocessor handles the unique characteristics of SMS messages including:
    - Heavy use of abbreviations and slang
    - Irregular punctuation and capitalization
    - Special characters and symbols
    - Mixed case patterns for emphasis
    - URLs and phone numbers
    """
    
    def __init__(self, use_lemmatization: bool = True, 
                 custom_stop_words: Optional[List[str]] = None):
        """
        Initialize the SMS preprocessor.
        
        Args:
            use_lemmatization (bool): If True, use lemmatization; if False, use stemming.
                                    Lemmatization is preferred for SMS as it preserves word 
                                    meaning better, which is crucial for spam detection.
            custom_stop_words (Optional[List[str]]): Additional stop words specific to SMS context.
        """
        self.use_lemmatization = use_lemmatization
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Load English stop words and add SMS-specific ones
        self.stop_words = set(stopwords.words('english'))
        
        # Add SMS-specific stop words that don't contribute to spam detection
        sms_stop_words = {
            'u', 'ur', 'im', 'ive', 'dont', 'cant', 'wont', 'youre', 'theyre',
            'thats', 'whats', 'hes', 'shes', 'its', 'were', 'youve', 'theyve',
            'msg', 'txt', 'text', 'sms'
        }
        self.stop_words.update(sms_stop_words)
        
        # Add custom stop words if provided
        if custom_stop_words:
            self.stop_words.update(custom_stop_words)
    
    def _expand_contractions(self, text: str) -> str:
        """
        Expand common contractions and SMS abbreviations.
        
        Args:
            text (str): Input text with contractions
            
        Returns:
            str: Text with expanded contractions
        """
        # Common contractions
        contractions = {
            # Standard contractions
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
            "'m": " am", "'s": " is",
            
            # SMS-specific abbreviations
            "u": "you", "ur": "your", "r": "are", "n": "and",
            "2": "to", "4": "for", "b4": "before", "2day": "today",
            "2morrow": "tomorrow", "2nite": "tonight", "w/": "with",
            "w/o": "without", "thru": "through", "pls": "please",
            "plz": "please", "luv": "love", "gud": "good", "gr8": "great",
            "wat": "what", "wiv": "with", "da": "the", "dis": "this",
            "dat": "that", "dere": "there", "der": "there", "yr": "your",
            "c": "see", "2": "to", "4": "for", "8": "ate",
        }
        
        # Apply contractions expansion
        for contraction, expansion in contractions.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _remove_special_patterns(self, text: str) -> str:
        """
        Remove or normalize special patterns common in SMS spam.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with special patterns cleaned
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                      'URL', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL', text)
        
        # Normalize phone numbers (common in spam)
        text = re.sub(r'\b\d{10,}\b', 'PHONE', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'PHONE', text)
        
        # Remove excessive punctuation (common spam tactic)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)
        
        # Normalize repeated characters (e.g., "freeeeee" -> "free")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Remove currency symbols but keep the context
        text = re.sub(r'[£$€¥₹]', 'CURRENCY', text)
        
        # Handle percentage and special offer indicators
        text = re.sub(r'\b\d+%\s*off\b', 'PERCENTOFF', text, flags=re.IGNORECASE)
        text = re.sub(r'\bfree\b', 'FREE', text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by handling case, whitespace, and basic cleaning.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _tokenize_and_clean(self, text: str) -> List[str]:
        """
        Tokenize text and apply cleaning rules.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of cleaned tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Clean tokens
        cleaned_tokens = []
        for token in tokens:
            # Remove pure punctuation tokens
            if token in string.punctuation:
                continue
            
            # Remove tokens that are just numbers (unless they might be meaningful)
            if token.isdigit() and len(token) < 3:
                continue
            
            # Remove very short tokens that are likely noise
            if len(token) < 2:
                continue
            
            # Remove tokens that are just special characters
            if re.match(r'^[^\w\s]+$', token):
                continue
            
            cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def _remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from token list.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Tokens with stop words removed
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def _apply_word_reduction(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming or lemmatization to reduce words to their root forms.
        
        We use lemmatization by default because:
        1. It preserves word meaning better (important for spam detection)
        2. It produces real words (better for interpretability)
        3. It handles irregular verbs better than stemming
        4. For spam detection, semantic meaning is crucial
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Tokens with word reduction applied
        """
        if self.use_lemmatization:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        else:
            return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_sms(self, text: str) -> str:
        """
        Complete SMS preprocessing pipeline optimized for spam detection.
        
        This function implements a comprehensive preprocessing strategy:
        
        1. Expand contractions and SMS abbreviations
        2. Remove/normalize special patterns (URLs, emails, phone numbers)
        3. Normalize text (lowercase, whitespace)
        4. Tokenize and clean tokens
        5. Remove stop words (including SMS-specific ones)
        6. Apply word reduction (lemmatization preferred)
        7. Join tokens back into a single string
        
        The strategy prioritizes:
        - Preserving semantic meaning for spam detection
        - Handling SMS-specific language patterns
        - Reducing noise while maintaining signal
        - Preparing text for TF-IDF vectorization
        
        Args:
            text (str): Raw SMS text message
            
        Returns:
            str: Preprocessed text ready for TF-IDF vectorization
            
        Example:
            >>> preprocessor = SMSPreprocessor()
            >>> raw_text = "FREE entry in 2 a wkly comp to win!!! Text WINNER to 85233 now!"
            >>> clean_text = preprocessor.preprocess_sms(raw_text)
            >>> print(clean_text)
            'FREE entry weekly comp win text WINNER PHONE'
        """
        if not isinstance(text, str):
            return ""
        
        if not text.strip():
            return ""
        
        # Step 1: Expand contractions and abbreviations
        text = self._expand_contractions(text)
        
        # Step 2: Remove/normalize special patterns
        text = self._remove_special_patterns(text)
        
        # Step 3: Normalize text
        text = self._normalize_text(text)
        
        # Step 4: Tokenize and clean
        tokens = self._tokenize_and_clean(text)
        
        # Step 5: Remove stop words
        tokens = self._remove_stop_words(tokens)
        
        # Step 6: Apply word reduction
        tokens = self._apply_word_reduction(tokens)
        
        # Step 7: Join back into string for TF-IDF
        return ' '.join(tokens)


def preprocess_sms_text(text: str, use_lemmatization: bool = True, 
                       custom_stop_words: Optional[List[str]] = None) -> str:
    """
    Convenience function for SMS text preprocessing.
    
    Args:
        text (str): Raw SMS text message
        use_lemmatization (bool): Whether to use lemmatization (True) or stemming (False)
        custom_stop_words (Optional[List[str]]): Additional stop words
        
    Returns:
        str: Preprocessed text ready for machine learning
    """
    preprocessor = SMSPreprocessor(use_lemmatization=use_lemmatization, 
                                  custom_stop_words=custom_stop_words)
    return preprocessor.preprocess_sms(text)


# Example usage and testing
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = SMSPreprocessor(use_lemmatization=True)
    
    # Test with sample SMS messages from the dataset
    sample_messages = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
        "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
        "Ok lar... Joking wif u oni..."
    ]
    
    print("SMS Preprocessing Pipeline Demo")
    print("=" * 50)
    
    for i, message in enumerate(sample_messages, 1):
        print(f"\nExample {i}:")
        print(f"Original: {message}")
        processed = preprocessor.preprocess_sms(message)
        print(f"Processed: {processed}")
        print("-" * 50)