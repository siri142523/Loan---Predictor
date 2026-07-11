# 🏦 Loan Approval Predictor (ML + Flask + Authentication)

A machine-learning-based Loan Approval System with **user authentication**,  
**Risk Score prediction**, and an **interactive dashboard** built using Flask.

---

## 📌 Overview

This project leverages machine learning models trained on a synthetic dataset to:

- Predict a **Risk Score** for loan applicants using **LGBM Regressor**
- Predict whether a **Loan will be Approved/Rejected** using **Random Forest Classifier**
- Display **prediction history & interactive charts** for insights
- Provide a **secure login/signup system** for users

The web interface is built using **Flask, HTML, CSS**, and includes **user authentication** for restricted access.

---

## 🔐 User Authentication (NEW)

A complete authentication system is added:

- **Login page**
- **Signup page**
- **Session-based login**
- **Protected dashboard (index.html)**
- Users stored in `users.json`

Only logged-in users can access:
- Prediction Form  
- Dashboard  
- Results  

### 🔑 Authentication Screenshots

#### 🔵 Login Page  
![Loan Approval Screenshot](https://github.com/siri142523/Loan---Predictor/raw/main/static/login.png)
#### 🟣 Signup Page  
![Loan Approval Screenshot](https://github.com/siri142523/Loan---Predictor/raw/main/static/signup.png)
#### 🟢 Dashboard  
![Loan Approval Screenshot](https://github.com/siri142523/Loan---Predictor/raw/main/static/dashboard.png)

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
├──static/                   # Static assets (CSS, images)
│   ├── style.css
│   ├── login.png
│   ├── signup.png
│   ├── dashboard.png
|
├── templates/                # Flask HTML templates
│   ├── index.html
│   ├── result.html
│   ├── login.html            
│   ├── signup.html           
│
├── app.py                    # Flask backend
├── .gitignore
├── requirements.txt
├── Loan.csv                  # Dataset
├── EDA.ipynb                 # Data analysis & feature engineering
└── model.ipynb               # Model training & selection

````

## 🔧 Data & Modeling

All preprocessing, feature engineering, and model experimentation are documented in:

- **EDA.ipynb** → Cleaning, visualization, correlation, feature engineering  
- **model.ipynb** → Model training & evaluation

### ✔ Final ML Pipeline

- Features standardized and numeric columns log-transformed (`log1p`)
- **LGBM Regressor** → Predicts Risk Score
- **Random Forest Classifier** → Predicts Loan Approval based on Risk Score
- Saved models:
  - `reg_model.pkl`
  - `clf_model.pkl`
  - `scaler.pkl`

---

## 🚀 Deployment

The Flask application provides:

### 📂 **Templates**
- **index.html** → Input form, progress bar, visual dashboard  
- **result.html** → Shows score, approval result, pie charts  
- **login.html** → Login UI  
- **signup.html** → Signup UI  

### 📁 **Backend**
- **app.py**
  - Routes for login, signup, prediction
  - Session authentication
  - Returns history + chart data as JSON

### 🎨 **Static**
- **style.css**  
- **login.png, signup.png, dashboard.png** (screenshots)

---

## 🧠 Results Summary

- **Risk Score Prediction:** LGBM Regressor with standardized features  
- **Loan Approval Prediction:** Random Forest Classifier  
- **Dashboard:** Progress bar, history table, pie charts  
- **Authenticated Access:** Only logged-in users can use the tool

---

## 💻 Future Improvements

- Save prediction history in a database.
- Add more visual analytics for loan trends.
- Deploy on Heroku or AWS for live access.
- Add password hashing for higher security  

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

