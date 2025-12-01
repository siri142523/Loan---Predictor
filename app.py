from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

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

# Serve the dashboard
@app.route('/')
def index():
    return render_template('index.html')


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
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


if __name__ == '__main__':
    app.run(debug=True)
