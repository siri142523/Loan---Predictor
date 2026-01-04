from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3

app = Flask(__name__)
app.secret_key = "loan_secret_key"
DB_FILE = "loan.db"

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
        session["loan_data"] = {
            "name": request.form["name"],
            "age": request.form["age"],
            "email": request.form["email"],
            "loan_type": request.form["loan_type"],
            "income": request.form["income"],
            "loan_amount": request.form["loan_amount"],
            "employment": request.form["employment"]
        }
        return redirect(url_for("verify"))

    return render_template("index.html", user=session["user"])

# ---------------- Verify ----------------
@app.route("/verify", methods=["GET","POST"])
def verify():
    if "loan_data" not in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        session["verification"] = {
            "aadhaar": "Verified",
            "pan": "Verified",
            "loan_history": "No previous loans"
        }
        return redirect(url_for("result"))

    return render_template("verify.html", user=session["user"])

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
    app.run(debug=True)
