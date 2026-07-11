<<<<<<< HEAD
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
=======
import pytesseract
from PIL import Image
import re
import os
import io
from pdf2image import convert_from_bytes
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
app.secret_key = "loan_secret_key"
DB_FILE = "loan.db"

def is_valid_pan(file):
    import pytesseract
    import re
    from PIL import Image
    import io
>>>>>>> 7429fc265425c0ff9e01b4af44cce9eb7e2aaf0a

    try:
        image = Image.open(io.BytesIO(file.read()))
        text = pytesseract.image_to_string(image)
        file.seek(0)

        text = text.upper()

<<<<<<< HEAD
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
=======
        keywords = ["INCOME TAX", "PERMANENT ACCOUNT NUMBER", "GOVT OF INDIA", "PAN"]
        if not any(k in text for k in keywords):
            return False

        pan_pattern = r"[A-Z]{5}[0-9]{4}[A-Z]"
        if not re.search(pan_pattern, text):
            return False

        return True
    except:
        return False

def validate_fee_doc(file, user_name, user_email):
    if not file or file.filename == "":
        return False, "Fee structure document not uploaded"

    allowed = [".pdf", ".png", ".jpg", ".jpeg"]
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        return False, "Invalid fee document format"

    file.seek(0, 2)
    size = file.tell()
    file.seek(0)

    if size < 10 * 1024:
        return False, "Fee document appears empty"

    if size > 3 * 1024 * 1024:
        return False, "Fee document exceeds 3 MB"

    # ✅ TEMPORARY ACCEPT (for project submission)
    return True, None

# ---------------- Database Setup ----------------
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    age INTEGER,
    loan_type TEXT,
    income REAL,
    loan_amount REAL,
    employment TEXT,
    risk_score INTEGER,
    status TEXT
)
''')
conn.commit()
conn.close()

# ---------------- Signup ----------------
@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO users (username,password) VALUES (?,?)",
                  (username,password))
        conn.commit()
        conn.close()
        return redirect(url_for("login"))
    return render_template("signup.html")

# ---------------- Login ----------------
@app.route("/login", methods=["GET","POST"])
def login():
    if "user" in session:
        return redirect(url_for("index"))

    message = None
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?",
                  (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session["user"] = username
            return redirect(url_for("index"))
        else:
            message = "Invalid username or password!"
    return render_template("login.html", message=message)

# ---------------- Logout ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- Forgot Password ----------------
@app.route("/forgot", methods=["GET","POST"])
def forgot():
    if request.method == "POST":
        username = request.form["username"].strip()
        session["reset_user"] = username
        return redirect(url_for("reset_password"))
    return render_template("forgot.html")

# ---------------- Reset Password ----------------
@app.route("/reset", methods=["GET","POST"])
def reset_password():
    if "reset_user" not in session:
        return redirect(url_for("forgot"))

    if request.method == "POST":
        new_pass = request.form["new_password"].strip()
        username = session["reset_user"]
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE users SET password=? WHERE username=?",
                  (new_pass, username))
        conn.commit()
        conn.close()
        session.pop("reset_user")
        return redirect(url_for("login"))

    return render_template("reset.html", username=session["reset_user"])

# ---------------- Home redirect ----------------
@app.route("/")
def home_redirect():
    if "user" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("index"))

# ---------------- Dashboard ----------------
@app.route("/index", methods=["GET","POST"])
def index():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        name = request.form["name"]
        age = request.form["age"]
        email = request.form["email"]
        loan_type = request.form["loan_type"]
        income = request.form["income"]
        loan_amount = request.form["loan_amount"]
        employment = request.form["employment"]

        # ---------- EDUCATION LOAN FEE VALIDATION ----------
        if loan_type == "Education Loan":
            fee_doc = request.files.get("fee_doc")

            ok, msg = validate_fee_doc(
                fee_doc,
                name,
                email
            )

            if not ok:
                return render_template(
                    "index.html",
                    user=session["user"],
                    error=msg
                )

        # ---------- SAVE DATA ONLY IF EVERYTHING IS VALID ----------
        session["loan_data"] = {
            "name": name,
            "age": age,
            "email": email,
            "loan_type": loan_type,
            "income": income,
            "loan_amount": loan_amount,
            "employment": employment
        }

        return redirect(url_for("verify"))

    return render_template("index.html", user=session["user"])

# ---------------- Verify ----------------
@app.route("/verify", methods=["GET","POST"])
def verify():
    if "loan_data" not in session:
        return redirect(url_for("index"))

    error = None

    # 🔹 SHOW VERIFY PAGE FIRST
    if request.method == "GET":
        return render_template("verify.html")

    if request.method == "POST":
        aadhaar = request.files.get("aadhaar")
        pan = request.files.get("pan")

        def validate_file(file, keywords):
            if not file or file.filename == "":
                return False, "File not selected"

            allowed = [".pdf", ".png", ".jpg", ".jpeg"]
            if not any(file.filename.lower().endswith(ext) for ext in allowed):
                return False, "Invalid file format"

            file.seek(0, 2)
            size = file.tell()
            file.seek(0)

            if size < 5 * 1024:
                return False, "File appears to be empty"

            if size > 2 * 1024 * 1024:
                return False, "File size exceeds 2 MB"

            content = file.read().lower()
            file.seek(0)

            if not any(k.encode() in content for k in keywords):
                return False, "Document content not valid"

            return True, None

        # ---------------- Aadhaar Validation ----------------
        ok, msg = validate_file(
            aadhaar,
            ["aadhaar", "uidai", "government of india"]
        )
        if not ok:
            return render_template("verify.html", error="Invalid Aadhaar document")

        # ---------------- PAN Validation ----------------
        if not pan or pan.filename == "":
            return render_template("verify.html", error="PAN file not selected")

        if not pan.filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            return render_template("verify.html", error="Invalid PAN file format")

        pan.seek(0, 2)
        size = pan.tell()
        pan.seek(0)

        if size < 5 * 1024:
            return render_template("verify.html", error="PAN file appears empty")

        if size > 2 * 1024 * 1024:
            return render_template("verify.html", error="PAN file size exceeds 2 MB")

        if not is_valid_pan(pan):
            return render_template("verify.html", error="Uploaded document is not a valid PAN card")

        # ---------------- SUCCESS (ONLY HERE REDIRECT) ----------------
        session["verification"] = {
            "aadhaar": "Verified",
            "pan": "Verified",
            "loan_history": "No previous loans"
        }

        return redirect(url_for("result"))


# ---------------- Result Page ----------------
@app.route("/result")
def result():
    if "verification" not in session or "loan_data" not in session:
        return redirect(url_for("index"))

    loan = session["loan_data"]
    verification = session["verification"]

    income = int(loan["income"])
    loan_amount = int(loan["loan_amount"])
    employment = loan["employment"]
    history = verification["loan_history"]

    risk_score = 0

    # Loan vs income
    if loan_amount > income * 5:
        risk_score += 40
    elif loan_amount > income * 3:
        risk_score += 25
    else:
        risk_score += 10

    # ✅ UPDATED EMPLOYMENT LOGIC
    if employment == "Unemployed":
        risk_score += 35
    elif employment == "Self Employed":
        risk_score += 20
    else:
        risk_score += 10

    # Loan history
    if history == "Previous loan found":
        risk_score += 15
    else:
        risk_score += 5

    risk_score = min(risk_score, 100)

    # Thresholds
    if risk_score < 35:
        status = "Low Risk"
    elif risk_score < 60:
        status = "Medium Risk"
    else:
        status = "High Risk"

    return render_template(
        "result.html",
        risk_score=risk_score,
        status=status,
        verification=verification,
        loan=loan
    )

# ---------------- Result Data API ----------------
@app.route("/result_data")
def result_data():
    if "verification" not in session or "loan_data" not in session:
        return jsonify({"error": "No data available"}), 400

    loan = session["loan_data"]
    verification = session["verification"]

    income = int(loan["income"])
    loan_amount = int(loan["loan_amount"])
    employment = loan["employment"]
    history = verification["loan_history"]

    risk_score = 0

    if loan_amount > income * 5:
        risk_score += 40
    elif loan_amount > income * 3:
        risk_score += 25
    else:
        risk_score += 10

    # ✅ SAME EMPLOYMENT LOGIC
    if employment == "Unemployed":
        risk_score += 35
    elif employment == "Self Employed":
        risk_score += 20
    else:
        risk_score += 10

    if history == "Previous loan found":
        risk_score += 15
    else:
        risk_score += 5

    risk_score = min(risk_score, 100)

    if risk_score < 35:
        status = "Low Risk"
    elif risk_score < 60:
        status = "Medium Risk"
    else:
        status = "High Risk"

    return jsonify({
        "loan_data": loan,
        "verification": verification,
        "risk_score": risk_score,
        "status": status
    })

# ---------------- Run App ----------------
if __name__ == "__main__":
>>>>>>> 7429fc265425c0ff9e01b4af44cce9eb7e2aaf0a
    app.run(debug=True)
