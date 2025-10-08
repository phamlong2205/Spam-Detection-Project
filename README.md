# 🚀 SMS & Email Spam Detection System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn 1.5+](https://img.shields.io/badge/scikit--learn-1.5+-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning system for detecting spam in SMS messages and emails using natural language processing and classical machine learning techniques. This project implements multiple algorithms with overfitting prevention, comprehensive feature engineering, and statistical analysis.

## 🎯 Key Features

### 🤖 **Machine Learning Models**
- **Classical ML**: SVM, Random Forest, Logistic Regression, Naive Bayes  
- **Overfitting Prevention**: SMOTE balancing, feature selection, cross-validation
- **Model Persistence**: Complete model serialization with preprocessing pipelines

### 📊 **Data Analysis & Visualization**
- **Performance Metrics**: Accuracy (94-97%), Precision, Recall, F1-Score
- **Statistical Testing**: Mann-Whitney U, Chi-square, Kolmogorov-Smirnov tests
- **Professional Visualizations**: Distribution plots, correlation analysis, ROC curves
- **Model Comparison**: Side-by-side performance benchmarking

### 🔧 **Feature Engineering Pipeline**
- **Text Processing**: Two-stage preprocessing (basic cleaning + advanced SMS handling)
- **Engineered Features**: 20+ features including message length, digit ratios, urgency indicators
- **Multi-source Support**: SMS and email datasets with different preprocessing strategies
- **Data Quality**: Automated filtering and deduplication

## 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **SVM** | **96.85%** | 96.40% | 95.85% | **96.12%** | 107.8s |
| **Random Forest** | 94.56% | **94.06%** | 93.66% | 93.86% | 1.2s |

*Results based on comprehensive evaluation with stratified cross-validation*

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.8 or higher**
- **Anaconda or Miniconda** (for environment management)

### Quick Setup with Anaconda
```bash
# Clone the repository
git clone https://github.com/phamlong2205/Spam-Detection-Project
cd spam-detection-project

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate spam-detection

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Alternative: Manual Setup
```bash
# Create conda environment
conda create -n spam-detection python=3.10 -y
conda activate spam-detection

# Install packages
conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn nltk -y
pip install imbalanced-learn beautifulsoup4
```

## 📁 Project Structure

```
spam-detection-project/
├── 📊 data/                              # Datasets and processed data
│   ├── spam.csv                          # Original SMS spam dataset
│   ├── emails.csv                        # Email spam dataset
│   ├── combined_messages.csv             # Merged dataset
│   ├── spam_with_features.csv            # Dataset with engineered features
│   └── spam_with_features_clean.csv      # Final cleaned dataset
│
├── 🤖 ML Pipeline Files/                  # Core model implementations  
│   ├── improved_model_pipeline.py        # Main training pipeline with overfitting prevention
│   ├── feature_engineering_pipeline.py   # Basic feature engineering (used in pipeline)
│   ├── sms_preprocessor.py              # Advanced SMS preprocessing class (standalone)
│   ├── demo_preprocessing.py            # Dataset normalization (SMS/email formats)
│   └── process_full_dataset.py          # Full data processing workflow
│
├── 📈 visualization/                     # Data analysis and visualization
│   └── simple_visualization_pipeline.py  # Statistical analysis and professional plots
│
├── 💾 saved_models_*/                    # Trained model artifacts
│   ├── svm_model.joblib                 # Serialized SVM model
│   ├── random_forest_model.joblib       # Serialized Random Forest model
│   ├── tfidf_vectorizer.joblib          # Text vectorizer
│   ├── feature_selector.joblib          # Feature selector
│   ├── standard_scaler.joblib           # Feature scaler
│   ├── label_encoder.joblib             # Label encoder
│   ├── model_metadata.json              # Model configuration
│   └── README_model_loading.md          # Model loading instructions
│
├── 🧪 tests/                            # Unit and integration tests
├── 📚 docs/                             # Documentation
├── 🔧 config/                           # Configuration files
├── 📝 examples/                         # Usage examples
├── 📋 requirements.txt                  # Python dependencies
└── 📖 README.md                         # This file
```

## 🚀 Quick Start

### Complete Pipeline (Recommended)
```bash
# Step 1: Normalize raw datasets (SMS spam.csv, email datasets)
python demo_preprocessing.py

# Step 2: Combine datasets and engineer features  
python process_full_dataset.py

# Step 3: Train all models with overfitting prevention
python improved_model_pipeline.py

# Step 4: Generate statistical analysis and visualizations
python simple_visualization_pipeline.py
```

### Load Pre-trained Models
```python
import joblib

# Load the best performing model (SVM - 96.85% accuracy)
svm_model = joblib.load('saved_models_TIMESTAMP/svm_model.joblib')
tfidf_vectorizer = joblib.load('saved_models_TIMESTAMP/tfidf_vectorizer.joblib')
feature_selector = joblib.load('saved_models_TIMESTAMP/feature_selector.joblib')
scaler = joblib.load('saved_models_TIMESTAMP/standard_scaler.joblib')
label_encoder = joblib.load('saved_models_TIMESTAMP/label_encoder.joblib')

# Load Random Forest model (94.56% accuracy)
rf_model = joblib.load('saved_models_TIMESTAMP/random_forest_model.joblib')
```

## 📊 Usage Examples

### Preprocessing Pipeline Understanding
```python
# The project uses a two-stage preprocessing approach:

# 1. Basic preprocessing (used in training pipeline)
from feature_engineering_pipeline import preprocess_text
basic_cleaned = preprocess_text("FREE entry! Call 123-456-7890 now!")
print(basic_cleaned)  # "free entri call now"

# 2. Advanced SMS preprocessing (standalone class)  
from sms_preprocessor import SMSPreprocessor
preprocessor = SMSPreprocessor()
advanced_cleaned = preprocessor.preprocess_sms("FREE entry! Call 123-456-7890 now!")
print(advanced_cleaned)  # "FREE entry call PHONE"
```

### Using Trained Models
```python
import joblib
import pandas as pd

# Load saved model and preprocessors (use same preprocessing as training)
model = joblib.load('saved_models_TIMESTAMP/svm_model.joblib')
vectorizer = joblib.load('saved_models_TIMESTAMP/tfidf_vectorizer.joblib')
scaler = joblib.load('saved_models_TIMESTAMP/standard_scaler.joblib')

# For new messages, use the same preprocessing as training pipeline
from feature_engineering_pipeline import preprocess_text

message = "URGENT! You've won $1000! Click here now!"
cleaned = preprocess_text(message)
tfidf_features = vectorizer.transform([cleaned])

# Get prediction
prediction = model.predict(tfidf_features)[0]
probability = model.predict_proba(tfidf_features)[0]

print(f"Message: {message}")
print(f"Classification: {'SPAM' if prediction == 1 else 'HAM'}")
print(f"Confidence: {probability[prediction]:.3f}")
```

### Feature Engineering Pipeline
```python
# The actual pipeline combines text preprocessing with numerical features
from feature_engineering_pipeline import apply_comprehensive_feature_engineering

# This function handles the complete feature engineering process
df = pd.DataFrame({'message': ['Free money!', 'Hi how are you?'], 
                   'label': ['spam', 'ham']})

# Apply same feature engineering as training
enhanced_df = apply_comprehensive_feature_engineering(df)

# View all engineered features  
print(enhanced_df.columns.tolist())
# Includes: cleaned_message, message_length, digit_ratio, capital_ratio, 
# special_char_count, urgency_words, financial_words, etc.
```

## ⚠️ Important Pipeline Notes

### Preprocessing Consistency
The project has **two different preprocessing approaches**:

1. **Training Pipeline** (`feature_engineering_pipeline.py`):
   - Basic text cleaning: lowercase, punctuation removal, stemming
   - Used in `improved_model_pipeline.py` for model training
   - **This is what trained models expect for new predictions**

2. **Advanced SMS Processor** (`sms_preprocessor.py`): 
   - Sophisticated SMS-specific preprocessing with lemmatization
   - Handles contractions, abbreviations, special patterns
   - **Standalone utility - not used in main training pipeline**

**For predictions**: Use the same preprocessing method that was used during training.

### Data Flow
```
Raw Data (spam.csv, emails.csv)
    ↓
demo_preprocessing.py (normalize formats)
    ↓  
process_full_dataset.py (combine + basic feature engineering)
    ↓
improved_model_pipeline.py (train models)
    ↓
saved_models_TIMESTAMP/ (serialized models + preprocessors)
```

### Model Performance Context
- Models were trained on **basic preprocessing** (stemming-based)
- The **advanced SMS preprocessing** may give different results
- For production use, maintain preprocessing consistency

## 🔧 Technical Details

### Environment Variables
Set these for optimal performance:
```bash
export CUDA_VISIBLE_DEVICES=0        # GPU usage (if available)
export OPENBLAS_NUM_THREADS=4        # CPU optimization
```

### Dataset Support
- **SMS**: `spam.csv` with columns `v1` (label) and `v2` (message)
- **Email**: Various formats automatically detected by `demo_preprocessing.py`
- **Combined**: Automatic deduplication and format standardization

## 🧪 Validation

```bash
# Verify pipeline works end-to-end
python improved_model_pipeline.py

# Check data processing
python process_full_dataset.py

# Generate analysis  
python simple_visualization_pipeline.py
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

**Quick contribution steps:**
1. Fork the repository
2. Create a feature branch
3. Make your changes  
4. Test the pipeline still works
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Documentation**: Check the `/docs` folder for detailed API documentation
- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/yourusername/spam-detection-project/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/yourusername/spam-detection-project/discussions) for questions and community support

## 🎓 Citations

If you use this project in your research, please cite:

```bibtex
@software{spam_detection_system,
  title={Advanced SMS & Email Spam Detection System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/spam-detection-project}
}
```

## 📊 Datasets

This project works with:
- **SMS Spam Collection**: Standard UCI ML Repository format
- **Enron Email Dataset**: Processed email spam data
- **Custom Datasets**: Automatic format detection for CSV files

Files are processed through the pipeline to create:
- `data/combined_messages.csv`: Merged and deduplicated data
- `data/spam_with_features.csv`: With engineered features
- `data/spam_with_features_clean.csv`: Final cleaned dataset

---
