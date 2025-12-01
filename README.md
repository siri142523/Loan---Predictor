# 🏦 Risk Assessment & Loan Approval Prediction

A Flask-based web application that predicts a Risk Score (regression) and Loan Approval (classification) based on personal and financial data. The app provides an interactive dashboard with progress bars, predicted history, and visualizations (like pie charts).

![Loan Approval Screenshot](https://github.com/siri142523/Loan---Predictor/raw/main/static/loan_dashboard.png)
---

## 📌 Overview

This project leverages machine learning models trained on a synthetic dataset to:

Predict a Risk Score for loan applicants using LGBM Regressor.
Predict whether a Loan will be approved using Random Forest Classifier.
Display prediction history and interactive charts for better insights.

The web interface is built using Flask, HTML, and CSS for a professional, user-friendly experience.
---

## 📁 Project Structure

```
Loan-Approval-Predictor/
│
├── models/                   # Saved ML models and scaler
│   ├── clf_model.pkl
│   ├── reg_model.pkl
│   └── scaler.pkl
│
├── static/                   # Static assets
│   ├── style.css
│   ├── loan_dashboard.png    # Dashboard screenshot
│   └── result.html           # Results page template
│
├── templates/                # Flask templates
│   └── index.html
│
├── app.py                    # Flask backend
├── .gitignore
├── requirements.txt
├── Loan.csv                  # Dataset
├── EDA.ipynb                 # Data analysis & feature engineering
└── model.ipynb               # Model training & selection

````

## 🔧 Data & Modeling

All preprocessing, feature engineering, and model testing are documented in EDA.ipynb.

Final selected models are saved and used in model.ipynb and the deployed Flask app.

Features standardized and some numeric features transformed using log1p.

Models predict risk score first, then loan approval based on the score.

---

## 🚀 Deployment

The Flask app provides:

index.html: Input form for applicant data with a progress bar.
result.html: Displays the predicted risk score, approval status, and visualizations (history, pie charts).
app.py: Handles predictions and routing.
static/style.css: Custom styling for the web interface.

---

## 🧠 Results Summary

-Risk Score Prediction: LGBM Regressor with scaled features.
-Loan Approval Prediction: Random Forest Classifier.
-Professional Dashboard: Progress bars, history table, pie charts.
-Interactive Results Page: result.html displays predictions and visual summaries neatly.

---

## 💻 Future Improvements

Add user authentication for multiple applicants.
Save prediction history in a database.
Add more visual analytics for loan trends.
Deploy on Heroku or AWS for live access.

---

## 💻 How to Use This Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/loan-approval-predictor.git
cd loan-approval-predictor
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

Then open your browser and go to:
📍 `http://127.0.0.1:5000`

---

