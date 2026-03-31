import sqlite3

def create_connection():
    return sqlite3.connect("loan_app.db", check_same_thread=False)

def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS applications")

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
        credit_history INTEGER,
        prediction INTEGER,
        risk_level TEXT,
        fraud_flag INTEGER
    )
    """)

    conn.commit()
    conn.close()

def insert_application(data):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO applications (
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_term, property_area, income_stability,
        credit_history, prediction, risk_level, fraud_flag
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()

def get_all_data():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM applications")
    data = cursor.fetchall()

    conn.close()
    return data
