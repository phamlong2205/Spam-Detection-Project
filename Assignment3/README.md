# AI4Cyber — Spam Detection Web Application

## 📋 Assignment 3: Full-Stack Web Application with AI Integration

**Project Title:** Intelligent SMS & Email Spam Detection System  
**Course:** COS30049 - Computing Technology Innovation Project  
**Institution:** Swinburne University of Technology

---

## 🎯 Project Overview

This full-stack web application provides real-time spam detection for SMS and email messages using machine learning. The system features a modern React frontend, FastAPI backend, and a pre-trained Random Forest classifier achieving 94.56% accuracy.

### Key Features
- ✅ Real-time spam/ham classification with confidence scores
- ✅ Interactive data visualizations (charts, radar plots, heatmaps)
- ✅ Prediction history with CSV export capability
- ✅ Dark/Light theme support
- ✅ Responsive design (desktop, tablet, mobile)
- ✅ RESTful API with automatic documentation

---

## 🏗️ System Architecture

```
┌─────────────────────┐         HTTP/REST          ┌─────────────────────┐
│   React Frontend    │ ◄────── Axios ──────────► │   FastAPI Backend   │
│   (Port 3000)       │                            │   (Port 8000)       │
└─────────────────────┘                            └──────────┬──────────┘
                                                              │
                                                              ▼
                                                    ┌─────────────────────┐
                                                    │   ML Models         │
                                                    │   - Random Forest   │
                                                    │   - TF-IDF Vector   │
                                                    │   - Feature Select  │
                                                    └─────────────────────┘
```

---

## 📦 Prerequisites

### System Requirements
- **Operating System:** macOS, Linux, or Windows 10+
- **Python:** 3.8 or higher
- **Node.js:** 18.0 or higher
- **npm:** 8.0 or higher
- **RAM:** Minimum 4GB (8GB recommended)
- **Disk Space:** 2GB free space

### Verify Your Installation
```bash
# Check Python version
python --version
# or
python3 --version

# Check Node.js version
node --version

# Check npm version
npm --version
```

---

## 🚀 Installation & Setup

### Step 1: Navigate to Assignment3 Directory

```bash
cd Assignment3
```

### Step 2: Backend Setup

#### 2.1 Navigate to Backend Directory
```bash
cd Back
```

#### 2.2 Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Alternative: Using Conda**
```bash
# Create conda environment
conda create -n spamDetection python=3.10 -y

# Activate environment
conda activate spamDetection
```

#### 2.3 Install Backend Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

**Required Libraries (automatically installed):**
- `fastapi==0.115.2` - Web framework
- `uvicorn[standard]==0.30.6` - ASGI server
- `pydantic==2.9.2` - Data validation
- `scikit-learn` - Machine learning
- `joblib` - Model persistence
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scipy` - Scientific computing

#### 2.4 Verify Model Files
Ensure these files exist in `Back/model/`:
```
Back/model/
├── tfidf_vectorizer.joblib        # TF-IDF text vectorizer
├── feature_selector.joblib         # Feature selection model
└── random_forest_model.joblib      # Trained Random Forest classifier
```

⚠️ **Important:** If model files are missing, the backend will not start!

---

### Step 3: Frontend Setup

#### 3.1 Navigate to Frontend Directory
```bash
# From Back directory, go to Front
cd ../Front

# Or from Assignment3 root
cd Front
```

#### 3.2 Install Frontend Dependencies
```bash
# Install all npm packages (this may take 2-5 minutes)
npm install
```

**Required Libraries (automatically installed):**
- `react@19.2.0` - UI framework
- `react-router-dom@7.9.4` - Routing
- `axios@1.12.2` - HTTP client
- `chart.js@4.5.1` - Charts
- `react-chartjs-2@5.3.0` - React Chart wrapper
- `d3@7.9.0` - Advanced visualizations

---

## ▶️ Running the Application

### You Need TWO Terminal Windows

#### Terminal 1: Start Backend Server

```bash
# Navigate to Back directory
cd Assignment3/Back

# Activate virtual environment (if using)
source venv/bin/activate
# or
conda activate spamDetection

# Start FastAPI server
uvicorn main:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

✅ **Backend is ready when you see:** `Application startup complete`

**Verify Backend:**
- Open browser: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

#### Terminal 2: Start Frontend Server

```bash
# Open a NEW terminal window/tab
# Navigate to Front directory
cd Assignment3/Front

# Start React development server
npm start
```

**Expected Output:**
```
Compiled successfully!

You can now view frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

✅ **Frontend is ready when you see:** `Compiled successfully!`

**The browser should automatically open:** http://localhost:3000

---

## 🧪 Testing the Application

### 1. Basic Functionality Test

1. **Open Frontend:** http://localhost:3000
2. **Navigate to Test Page:** Click "Try the Tester →"
3. **Enter Test Messages:**
   
   **Spam Example:**
   ```
   FREE! You've won $1000! Click here NOW to claim your prize! Call 555-0123
   ```
   
   **Ham Example:**
   ```
   Hey, are we still meeting for coffee tomorrow at 3pm?
   ```

4. **Click "Predict (RF)"**
5. **Verify Results:**
   - Badge shows "Spam" or "Ham"
   - Confidence percentage displayed
   - Doughnut chart appears
   - Entry added to history table

### 2. History Management Test

- **View History:** Scroll down to history table
- **Change Limit:** Select different limits (10/20/30/50/100)
- **Refresh:** Click "Refresh" button
- **Export:** Click "Export CSV" to download predictions
- **Delete Entry:** Click "Delete" on any row
- **Clear All:** Click "Clear All" to remove all history

### 3. API Endpoint Test

```bash
# Test prediction endpoint
curl -X POST "http://localhost:8000/predict/save" \
  -H "Content-Type: application/json" \
  -d '{"text":"FREE money now!"}'

# Test history endpoint
curl "http://localhost:8000/history?limit=5"

# Test health endpoint
curl "http://localhost:8000/health"
```

---

## 📖 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Root Endpoint
- **URL:** `/`
- **Method:** `GET`
- **Response:**
```json
{
  "message": "Spam Detection API (RF only)",
  "version": "3.0.0"
}
```

#### 2. Health Check
- **URL:** `/health`
- **Method:** `GET`
- **Response:**
```json
{
  "model": "rf",
  "vectorizer_features": 5000,
  "selector_output_features": 3000,
  "status": "ok",
  "classes": ["ham", "spam"]
}
```

#### 3. Make Prediction (Main Endpoint)
- **URL:** `/predict/save`
- **Method:** `POST`
- **Request Body:**
```json
{
  "text": "Your message here"
}
```
- **Response:**
```json
{
  "model": "rf",
  "label": "spam",
  "label_num": 1,
  "prob": 0.8534,
  "threshold": 0.50,
  "length": 25
}
```

#### 4. Get History
- **URL:** `/history?limit=20`
- **Method:** `GET`
- **Query Parameters:**
  - `limit` (optional, default=20): Number of records to return
- **Response:**
```json
{
  "count": 20,
  "items": [...]
}
```

#### 5. Delete History Entry
- **URL:** `/history/{index}`
- **Method:** `DELETE`
- **Response:**
```json
{
  "message": "Item deleted",
  "deleted": {...}
}
```

#### 6. Clear All History
- **URL:** `/history`
- **Method:** `DELETE`
- **Response:**
```json
{
  "message": "History cleared",
  "deleted_count": 50
}
```

#### 7. Export Predictions
- **URL:** `/export/predictions?format=csv`
- **Method:** `GET`
- **Query Parameters:**
  - `format`: `csv` or `json`
- **Response:** CSV file download or JSON array

### Interactive API Documentation
Visit **http://localhost:8000/docs** for Swagger UI with live testing capability.

---

## 🤖 AI Model Integration

### Model Architecture

**Primary Model:** Random Forest Classifier
- **Algorithm:** Ensemble of 100 decision trees
- **Training Accuracy:** 94.56%
- **Features:** TF-IDF vectors + feature selection
- **Threshold:** 0.50 for spam classification

### Model Pipeline

```
Input Text
    ↓
TF-IDF Vectorization (1-2 grams, max 5000 features)
    ↓
Feature Selection (SelectKBest with chi2)
    ↓
Random Forest Prediction
    ↓
Probability Calculation P(spam)
    ↓
Threshold Application (≥0.50 = Spam)
    ↓
Output: {label, probability, confidence}
```

### Model Files Explanation

1. **`tfidf_vectorizer.joblib`**
   - Converts text to numerical TF-IDF features
   - Vocabulary size: ~5000 words
   - N-grams: (1,2) - captures single words and word pairs

2. **`feature_selector.joblib`**
   - Selects most discriminative features using chi-square test
   - Reduces dimensionality for better performance
   - Keeps top 3000 features

3. **`random_forest_model.joblib`**
   - Pre-trained Random Forest classifier
   - 100 decision trees with max depth optimization
   - Balanced class weights

---

## 🎨 Frontend Pages

### 1. Home Page (`/`)
- Project introduction and overview
- Feature highlights
- Navigation to Test and About pages

### 2. Test Page (`/test`)
- **Main prediction interface**
- Text input with validation (3-1000 characters)
- Real-time classification results
- Confidence visualization (doughnut chart)
- Prediction history table
- Export and management features

### 3. About Page (`/about`)
- Model explanation and training pipeline
- Team member profiles
- Interactive D3.js visualizations:
  - Feature comparison bar chart
  - Scatter plot analysis
  - Radar chart profiles
  - Correlation heatmap
  - Box plot distributions
- Training performance charts

### 4. Theme Toggle
- Light/Dark mode switcher in header
- Persists user preference in localStorage

---

## 🐛 Troubleshooting

### Backend Issues

#### Problem: "ModuleNotFoundError: No module named 'fastapi'"
**Solution:**
```bash
pip install -r requirements.txt
```

#### Problem: "FileNotFoundError: model/tfidf_vectorizer.joblib"
**Solution:** Ensure all model files are present in `Back/model/` directory

#### Problem: "Address already in use (port 8000)"
**Solution:**
```bash
# Kill process on port 8000 (macOS/Linux)
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn main:app --reload --port 8001
```

#### Problem: "uvicorn: command not found"
**Solution:**
```bash
pip install uvicorn[standard]
```

---

### Frontend Issues

#### Problem: "Port 3000 already in use"
**Solution:**
```bash
# Kill process on port 3000 (macOS/Linux)
lsof -ti:3000 | xargs kill -9

# Or use different port
PORT=3001 npm start
```

#### Problem: "npm ERR! missing script: start"
**Solution:**
```bash
cd Assignment3/Front  # Ensure you're in correct directory
npm install
npm start
```

#### Problem: Frontend won't compile or hangs
**Solution:**
```bash
# Clear cache and reinstall
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
npm start
```

#### Problem: "Network Error" when making predictions
**Solution:**
- Verify backend is running on http://localhost:8000
- Check browser console for errors
- Ensure both services are on correct ports

---

## 📁 Project Structure

```
Assignment3/
├── Back/                           # Backend (FastAPI)
│   ├── main.py                     # Main API server
│   ├── requirements.txt            # Python dependencies
│   ├── history.json               # Prediction history storage
│   └── model/                      # ML model files
│       ├── tfidf_vectorizer.joblib
│       ├── feature_selector.joblib
│       └── random_forest_model.joblib
│
├── Front/                          # Frontend (React)
│   ├── public/                     # Static assets
│   │   ├── index.html
│   │   └── assets/                 # Images & charts
│   ├── src/
│   │   ├── App.js                  # Main React component
│   │   ├── App.css                 # Global styles
│   │   ├── index.js               # Entry point
│   │   ├── pages/                  # Page components
│   │   │   ├── Home.js            # Landing page
│   │   │   ├── Test.js            # Prediction interface
│   │   │   ├── About.js           # Model info & viz
│   │   │   └── Dashboard.js       # Analytics
│   │   └── components/             # Reusable components
│   │       ├── Header.js
│   │       ├── Footer.js
│   │       └── Navbar.js
│   ├── package.json               # npm dependencies
│   └── package-lock.json
│
└── README.md                       # This file
```

---

## 📊 Performance Metrics

### Model Performance
- **Accuracy:** 94.56%
- **Precision:** 94.06%
- **Recall:** 93.66%
- **F1-Score:** 93.86%
- **Training Time:** 1.2 seconds

### System Performance
- **API Response Time:** <100ms average
- **Frontend Load Time:** <2s initial load
- **Concurrent Users:** Tested up to 10 simultaneous predictions

---

## 👥 Team Information

**Group Name:** AI4Cyber Team  
**Course:** COS30049 - Computing Technology Innovation Project

**Team Members:**
- **Duc Tri Tran** - Data/ML Lead, Project Manager
- **Quoc Phi Long Pham** - Web Development Lead  
- **Hengheng Lonh** - QA Lead

---

## ✅ Pre-Submission Checklist

Before submitting Assignment 3, ensure:

- [ ] Both backend and frontend run without errors
- [ ] All model files are present in `Back/model/`
- [ ] Prediction functionality works correctly
- [ ] History management features work (add/delete/export)
- [ ] All pages are accessible (Home, Test, About)
- [ ] Charts and visualizations display properly
- [ ] Theme toggle works (light/dark mode)
- [ ] API documentation accessible at `/docs`
- [ ] README is complete and accurate
- [ ] `node_modules` excluded from submission ZIP
- [ ] Demonstration video recorded (max 7 minutes)

---

## 🎥 Demo Video Guidelines

Your demonstration video should cover:

1. **Introduction** (30s) - Project overview and team
2. **Backend Demo** (1 min) - Starting server, API docs, health check
3. **Frontend Demo** (3 min) - All pages, making predictions, history
4. **Visualizations** (1.5 min) - Interactive charts on About page
5. **Responsive Design** (1 min) - Desktop/tablet/mobile views
6. **Technical Highlights** (1 min) - Challenges and solutions

---

## 📚 Additional Resources

### Documentation
- **FastAPI:** https://fastapi.tiangolo.com/
- **React:** https://react.dev/
- **Chart.js:** https://www.chartjs.org/
- **D3.js:** https://d3js.org/
- **scikit-learn:** https://scikit-learn.org/

---

**Last Updated:** November 2025  
**Version:** 3.0.0  
**Assignment:** COS30049 Assignment 3

---

## 🚀 Quick Start Commands

**Backend:**
```bash
cd Assignment3/Back
source venv/bin/activate  # or: conda activate spamDetection
uvicorn main:app --reload --port 8000
```

**Frontend (in new terminal):**
```bash
cd Assignment3/Front
npm start
```

**Access:**
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

