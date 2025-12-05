from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import numpy as np
import pandas as pd
import pickle
import json
import os

app = Flask(__name__)
app.secret_key = "your_secret_key_here"   # CHANGE THIS TO ANY RANDOM STRING

# --------------------------
# AUTHENTICATION SETUP
# --------------------------

USERS_FILE = 'users.json'

# Ensure the users.json file exists
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({"admin": "admin123", "testuser": "testuser123"}, f, indent=4)

def is_logged_in():
    return "username" in session

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        # Load fresh users every time
        with open(USERS_FILE, 'r') as f:
            USERS = json.load(f)

        # Validate credentials
        if username in USERS and USERS[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --------------------------
# EXISTING ORIGINAL CODE
# --------------------------

# Load models and scaler
with open('models/reg_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/clf_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)

# Mappings
status_map = {'Unemployed': 0, 'Self-Employed': 1, 'Employed': 2}
edu_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}

features_to_log1p = ['LoanAmount', 'MonthlyIncome', 'NetWorth']
num_cols_to_standardize = [
    'Age', 'CreditScore', 'LoanAmount', 'LoanDuration',
    'CreditCardUtilizationRate', 'LengthOfCreditHistory',
    'MonthlyIncome', 'NetWorth', 'InterestRate'
]

# In-memory prediction history
prediction_history = []

# Protect dashboard
@app.route('/')
def index():
    if not is_logged_in():
        return redirect(url_for('login'))
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json

    # Convert input to DataFrame
    input_df = pd.DataFrame([data])

    # Encode categorical features
    input_df['EmploymentStatus'] = input_df['EmploymentStatus'].map(status_map)
    input_df['EducationLevel'] = input_df['EducationLevel'].map(edu_map)

    # Log1p transform
    input_df[features_to_log1p] = input_df[features_to_log1p].apply(np.log1p)

    # Standardize numerical columns
    input_df[num_cols_to_standardize] = scaler.transform(input_df[num_cols_to_standardize])

    # Predict risk score
    risk_score = round(reg_model.predict(input_df)[0], 2)
    approval_status = clf_model.predict([[risk_score]])[0]
    approval_text = "Approved" if approval_status == 1 else "Rejected"

    # Save to history
    prediction_history.append({
        'RiskScore': risk_score,
        'ApprovalStatus': approval_text
    })

    # Prepare pie chart data
    df_hist = pd.DataFrame(prediction_history)
    pie_data = df_hist['ApprovalStatus'].value_counts().to_dict()

    # Return JSON
    return jsonify({
        'risk_score': risk_score,
        'approval_status': approval_text,
        'history': prediction_history,
        'pie_data': pie_data
    })

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Load existing users every time
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)

        # Check if user already exists
        if username in users:
            return "User already exists! Try a different username."

        # Add new user
        users[username] = password

        # Save back to file
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)

        return "Signup successful! Now go to Login page."

    return render_template('signup.html')

if __name__ == '__main__':
    app.run(debug=True)
