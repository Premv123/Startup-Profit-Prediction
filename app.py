from flask import Flask, request, render_template, redirect, url_for, jsonify, session, send_file, Response
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pickle
import json
import sqlite3
import matplotlib.pyplot as plt
import io
import os
import hashlib
from datetime import datetime
import random

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')

# Load model and columns once
with open('models/startup_profit_prediction_lr_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# -------------------------- DB INIT ----------------------------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    email TEXT,
                    phone_no INTEGER,
                    R_address TEXT,
                    gender TEXT,
                    age INTEGER,
                    dob DATE
                )''')
    conn.commit()
    conn.close()

init_db()

# ------------------------ PREDICT LOGIC -------------------------
def predict_profit(r_d_expenses, administration_expenses, marketing_expenses, state):
    x = np.zeros(len(data_columns))
    try:
        x[data_columns.index('r_d_expenses')] = float(r_d_expenses)
        x[data_columns.index('administration_expenses')] = float(administration_expenses)
        x[data_columns.index('marketing_expenses')] = float(marketing_expenses)

        state_col = f"state_{state.lower()}"
        if state_col in data_columns:
            x[data_columns.index(state_col)] = 1

        prediction = model.predict([x])[0]
        return round(prediction, 2)
    except Exception as e:
        return f"Prediction error: {e}"

# ------------------------ ROUTES -------------------------------

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['U_password']
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, hashed_password))
            user = c.fetchone()
            conn.close()

            if user:
                session['user'] = {
                    'id': user[0], 'username': user[1], 'email': user[3],
                    'phone_no': user[4], 'R_address': user[5], 'gender': user[6],
                    'age': user[7], 'dob': user[8]
                }
                return redirect('/info')
            else:
                return render_template('login1.html', error='Invalid username or password')
        except Exception:
            return render_template('error.html', message="An error occurred during login.")
    return render_template('login1.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            data = request.form
            password_hash = hashlib.sha256(data['U_password'].encode()).hexdigest()
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password_hash, email, phone_no, R_address, gender, age, dob) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                      (data['username'], password_hash, data['email'], data['phone_no'], data['R_address'], data['gender'], data['age'], data['dob']))
            conn.commit()
            conn.close()
            return "Registration successful!", 200
        except Exception as e:
            return f"Registration error: {e}", 500
    return render_template('register1.html')


@app.route('/info')
def info():
    user = session.get('user')
    return render_template("info.html", user=user) if user else redirect(url_for('login'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    user = session.get('user')
    if not user:
        return redirect(url_for('login'))

    if request.method == 'POST':
        form = request.form
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('''UPDATE users SET username=?, email=?, phone_no=?, R_address=?, gender=?, age=?, dob=? WHERE id=?''',
                      (form['username'], form['email'], form['phone_no'], form['R_address'], form['gender'], form['age'], form['dob'], user['id']))
            conn.commit()
            conn.close()
            session['user'].update(form)
            return redirect(url_for('profile'))
        except sqlite3.IntegrityError:
            return render_template('profile.html', user=user, error='Username already exists.')
        except Exception as e:
            return render_template('profile.html', user=user, error=f'An error occurred: {e}')

    return render_template('profile.html', user=user)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/government_schemes")
def government_schemes():
    return render_template('government_schemes.html')

@app.route("/help_desk")
def help_desk():
    return render_template('help_desk.html')


@app.route('/predict', methods=['POST'])
def predict():
    r_d_expenses = float(request.form['r_d_expenses'])
    administration_expenses = float(request.form['administration_expenses'])
    marketing_expenses = float(request.form['marketing_expenses'])
    state = request.form['state']

    predicted_profit = predict_profit(r_d_expenses, administration_expenses, marketing_expenses, state)
    return render_template('result.html', prediction=predicted_profit,
                           r_d_expenses=r_d_expenses,
                           administration_expenses=administration_expenses,
                           marketing_expenses=marketing_expenses,
                           state=state)


@app.route('/result')
def result():
    current_date = datetime.now().strftime("%B %d, %Y")
    reference_number = random.randint(10000, 99999)
    return render_template('result.html',
                           current_date=current_date,
                           reference_number=reference_number)


# ---------------------- GRAPH ROUTES ---------------------------

def generate_bar_graph(data):
    fig, ax = plt.subplots()
    categories = list(data.keys())
    values = list(data.values())
    ax.bar(categories, values)
    ax.set_title('Startup Expenses')
    ax.set_ylabel('Amount in $')
    ax.set_xlabel('Expense Category')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_pie_chart(data):
    fig, ax = plt.subplots()
    labels = list(data.keys())
    sizes = list(data.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.set_title('Startup Expenses Distribution')
    plt.axis('equal')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf


@app.route('/bar_plot')
def bar_plot():
    data = {
        'R&D': float(request.args.get('r_d_expenses', 0)),
        'Admin': float(request.args.get('administration_expenses', 0)),
        'Marketing': float(request.args.get('marketing_expenses', 0))
    }
    return Response(generate_bar_graph(data).getvalue(), mimetype='image/png')


@app.route('/pie_plot')
def pie_plot():
    data = {
        'R&D': float(request.args.get('r_d_expenses', 0)),
        'Admin': float(request.args.get('administration_expenses', 0)),
        'Marketing': float(request.args.get('marketing_expenses', 0))
    }
    return Response(generate_pie_chart(data).getvalue(), mimetype='image/png')


# -------------------- MAIN -------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
