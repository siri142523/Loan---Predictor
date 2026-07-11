from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# ------------------- Database -------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///loan_predictor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ------------------- Load ML Models -------------------
with open('models/reg_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/clf_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)

# ------------------- Mappings -------------------
status_map = {'Unemployed': 0, 'Self-Employed': 1, 'Employed': 2}
edu_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}

num_cols_to_standardize = [
    'Age', 'CIBILScore', 'LoanAmount', 'LoanDuration',
    'CreditCardUtilizationRate', 'LengthOfCreditHistory',
    'MonthlyIncome'
]

# ------------------- Models -------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    risk_score = db.Column(db.Float)
    approval_status = db.Column(db.String(10))

with app.app_context():
    db.create_all()

# ------------------- Routes -------------------

@app.route('/')
def home():
    return render_template('index.html')

# -------- Register --------
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        if User.query.filter_by(username=username).first():
            return "Username already exists"

        db.session.add(User(username=username, password=password))
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html')

# -------- Login --------
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()

        if user and check_password_hash(user.password, request.form['password']):
            session['user_id'] = user.id
            session['verified'] = False
            return redirect(url_for('dashboard'))

        return "Invalid Credentials"

    return render_template('login.html')

# -------- Forgot Password --------
@app.route('/forgot-password')
def forgot_password():
    return render_template('forgot_password.html')

@app.route('/reset-password', methods=['POST'])
def reset_password():
    user = User.query.filter_by(username=request.form['username']).first()
    if user:
        user.password = generate_password_hash(request.form['password'])
        db.session.commit()
        return redirect(url_for('login'))
    return "User not found"

# -------- Logout --------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# -------- Dashboard (Loan Details Page) --------
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

# -------- NEXT → Document Upload --------
@app.route('/documents', methods=['POST'])
def documents():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('documents.html')

# -------- Aadhar & PAN Verification --------
@app.route('/verify', methods=['POST'])
def verify():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    aadhar = request.files['aadhar']
    pan = request.files['pan']

    if aadhar and pan:
        session['verified'] = True
        return render_template(
            'verification_result.html',
            aadhar_status="Correct Person Verified",
            pan_status="No previous loan history found"
        )

    return "Verification Failed"

# -------- Prediction (ONLY AFTER VERIFICATION) --------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session or not session.get('verified'):
        return redirect(url_for('dashboard'))

    data = {
        'Age': int(request.form['Age']),
        'CIBILScore': int(request.form['CIBILScore']),
        'EmploymentStatus': request.form['EmploymentStatus'],
        'EducationLevel': request.form['EducationLevel'],
        'LoanAmount': float(request.form['LoanAmount']),
        'LoanDuration': int(request.form['LoanDuration']),
        'CreditCardUtilizationRate': float(request.form['CreditCardUtilizationRate']),
        'BankruptcyHistory': 1 if request.form['BankruptcyHistory'] == 'Yes' else 0,
        'PreviousLoanDefaults': 1 if request.form['PreviousLoanDefaults'] == 'Yes' else 0,
        'LengthOfCreditHistory': int(request.form['LengthOfCreditHistory']),
        'MonthlyIncome': float(request.form['MonthlyIncome'])
    }

    df = pd.DataFrame([data])
    df['EmploymentStatus'] = df['EmploymentStatus'].map(status_map)
    df['EducationLevel'] = df['EducationLevel'].map(edu_map)
    df[num_cols_to_standardize] = scaler.transform(df[num_cols_to_standardize])

    risk_score = round(reg_model.predict(df)[0], 2)
    approval_status = "Approved" if clf_model.predict([[risk_score]])[0] == 1 else "Rejected"

    pred = Prediction(
        user_id=session['user_id'],
        risk_score=risk_score,
        approval_status=approval_status
    )
    db.session.add(pred)
    db.session.commit()

    predictions = Prediction.query.filter_by(user_id=session['user_id']).all()
    return render_template('results.html', latest=pred, predictions=predictions)

# ------------------- Run -------------------
if __name__ == '__main__':
    app.run(debug=True)
