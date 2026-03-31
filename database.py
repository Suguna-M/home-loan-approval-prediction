import sqlite3

# -----------------------------
# CONNECT
# -----------------------------
def create_connection():
    return sqlite3.connect("loan_app.db", check_same_thread=False)

# -----------------------------
# CREATE TABLE
# -----------------------------
def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gender INTEGER,
        married INTEGER,
        dependents INTEGER,
        education INTEGER,
        self_employed INTEGER,
        applicant_income REAL,
        coapplicant_income REAL,
        loan_amount REAL,
        loan_term REAL,
        property_area INTEGER,
        income_stability INTEGER,
        prediction INTEGER,
        risk_level TEXT,
        fraud_flag INTEGER
    )
    """)

    conn.commit()
    conn.close()

# -----------------------------
# INSERT DATA
# -----------------------------
def insert_application(data):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO applications (
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_term, property_area, income_stability,
        prediction, risk_level, fraud_flag
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()

# -----------------------------
# FETCH DATA
# -----------------------------
def get_all_data():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM applications")
    data = cursor.fetchall()

    conn.close()
    return data
