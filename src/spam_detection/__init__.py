"""
SMS Spam Detection Package

This package provides tools for SMS spam detection with advanced text preprocessing.
"""

# Import the main classes so users can do: from spam_detection import SMSPreprocessor
from .preprocessing import SMSPreprocessor, preprocess_sms_text

# Define what gets imported with "from spam_detection import *"
__all__ = ['SMSPreprocessor', 'preprocess_sms_text']

# Package metadata
__version__ = "1.0.0"