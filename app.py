from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the models and scaler
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect form data
        input_data = {
            'Age': int(request.form['Age']),
            'CreditScore': int(request.form['CreditScore']),
            'EmploymentStatus': request.form['EmploymentStatus'],
            'EducationLevel': request.form['EducationLevel'],
            'LoanAmount': float(request.form['LoanAmount']),
            'LoanDuration': int(request.form['LoanDuration']),
            'CreditCardUtilizationRate': float(request.form['CreditCardUtilizationRate']),
            'BankruptcyHistory': int(request.form['BankruptcyHistory']),
            'PreviousLoanDefaults': int(request.form['PreviousLoanDefaults']),
            'LengthOfCreditHistory': int(request.form['LengthOfCreditHistory']),
            'MonthlyIncome': float(request.form['MonthlyIncome']),
            'NetWorth': float(request.form['NetWorth']),
            'InterestRate': float(request.form['InterestRate']),
        }

        input_df = pd.DataFrame([input_data])

        # Encoding categorical features
        input_df['EmploymentStatus'] = input_df['EmploymentStatus'].map(status_map)
        input_df['EducationLevel'] = input_df['EducationLevel'].map(edu_map)

        # Log1p transform
        input_df[features_to_log1p] = input_df[features_to_log1p].apply(np.log1p)

        # Standardize numerical columns
        input_df[num_cols_to_standardize] = scaler.transform(input_df[num_cols_to_standardize])

        # Predict risk score
        risk_score = round(reg_model.predict(input_df)[0], 2)
        approval_status = clf_model.predict([[risk_score]])[0]
        prediction_color = "green" if approval_status == 1 else "red"

        return render_template(
            'index.html',
            risk_score=risk_score,
            color=prediction_color
        )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
