import os
import csv
import sys
import random
import joblib
import yfinance as yf
import numpy as np
import pytz
from io import StringIO
from datetime import datetime, timedelta
from functools import wraps
import re

import pandas as pd
import psutil
import subprocess

from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin,
    login_user, logout_user, login_required, current_user
)
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
from sqlalchemy import func
from tensorflow.keras.models import load_model

from services.finnhub_api import (
    get_stock_quote, get_market_news, get_comparison_data,
    get_stock_history, get_trending_stocks, get_market_indices
)

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)

app.config['SECRET_KEY']                     = os.getenv("FLASK_SECRET", "dev_key_123")
app.config['SQLALCHEMY_DATABASE_URI']        = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ── Session & Cookie Security ─────────────────────────────────
# HttpOnly  → JS cannot read cookies (blocks XSS theft)
# SameSite  → Cookie not sent on cross-origin requests (blocks CSRF)
app.config['PERMANENT_SESSION_LIFETIME']  = timedelta(hours=1)
app.config['SESSION_REFRESH_EACH_REQUEST'] = True
app.config['REMEMBER_COOKIE_DURATION']    = timedelta(days=7)
app.config['SESSION_COOKIE_HTTPONLY']     = True
app.config['REMEMBER_COOKIE_HTTPONLY']    = True
app.config['SESSION_COOKIE_SAMESITE']     = 'Lax'
app.config['REMEMBER_COOKIE_SAMESITE']    = 'Lax'
app.config['SESSION_COOKIE_SECURE']       = False  
app.config['REMEMBER_COOKIE_SECURE']      = False  

db            = SQLAlchemy(app)
bcrypt        = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view             = 'login'
login_manager.login_message_category = 'info'

WINDOW_SIZE   = 60                    
TRAINING_LOCK = "training.lock"       # Lock-file path — exists while training runs


# ─────────────────────────────────────────────
# DATABASE MODELS
# ─────────────────────────────────────────────
class User(db.Model, UserMixin):
    id          = db.Column(db.Integer, primary_key=True)
    full_name   = db.Column(db.String(100), nullable=False)
    email       = db.Column(db.String(120), unique=True, nullable=False)
    password    = db.Column(db.String(200), nullable=False)
    role        = db.Column(db.String(20),  default='USER')
    created_at  = db.Column(db.DateTime,    default=db.func.current_timestamp())
    predictions = db.relationship(
        'Prediction', backref='owner', cascade="all, delete-orphan"
    )


class Prediction(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    symbol         = db.Column(db.String(20), nullable=False)
    price          = db.Column(db.Float,      nullable=False)
    recommendation = db.Column(db.String(20), nullable=False)
    user_id        = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at     = db.Column(db.DateTime, default=db.func.current_timestamp())


class AuditLog(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120))
    action     = db.Column(db.String(255))
    timestamp  = db.Column(db.DateTime, default=db.func.current_timestamp())

# ─────────────────────────────────────────────
# VALIDATION HELPERS
# ─────────────────────────────────────────────
def is_valid_email(email):
    """Checks for a standard email format."""
    regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w+$'
    return re.search(regex, email)

def validate_password_strength(password):
    """
    Constraints:
    - Min 8 characters
    - At least 1 uppercase letter
    - At least 1 number
    - At least 1 special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not any(char.isupper() for char in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one number."
    if not any(char in "!@#$%^&*()-_=+[]{}|;:,.<>?/" for char in password):
        return False, "Password must contain at least one special character."
    return True, ""
# ─────────────────────────────────────────────
# TRAINING LOCK HELPERS
# ─────────────────────────────────────────────
def is_training_running() -> bool:
    """
    Returns True if the lock file exists AND the PID inside it
    belongs to a process that is still alive.

    Why check the PID?
    If the server crashed mid-training, the lock file would be left
    behind (a "stale lock"). Without the PID check, training could
    never be started again until someone manually deletes the file.
    The PID check makes the lock self-healing.
    """
    if not os.path.exists(TRAINING_LOCK):
        return False

    try:
        with open(TRAINING_LOCK, 'r') as f:
            pid = int(f.read().strip())
        return psutil.pid_exists(pid)
    except (ValueError, IOError):
        # Lock file is corrupt or unreadable → treat as stale, remove it
        _remove_lock()
        return False


def _remove_lock():
    """Silently remove the lock file if it exists."""
    try:
        os.remove(TRAINING_LOCK)
    except FileNotFoundError:
        pass


# ─────────────────────────────────────────────
# GENERAL HELPERS
# ─────────────────────────────────────────────
def _write_log(email: str, action: str):
    """Single DB write point for all audit logs."""
    try:
        db.session.add(AuditLog(user_email=email, action=action))
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"[AuditLog Error] {e}")


def log_action(action: str):
    """Log an action for the currently authenticated user."""
    email = current_user.email if current_user.is_authenticated else "Guest/System"
    _write_log(email, action)


def log_action_for(email: str, action: str):
    """
    Log an action for a specific email address.
    Use this right after login_user() when current_user is not yet set.
    """
    _write_log(email, action)


def admin_required(f):
    """
    Decorator: ensures user is authenticated AND has ADMIN role.
    Always place AFTER @login_required so current_user is guaranteed set.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'ADMIN':
            flash("Access Denied: Admin privileges required.", "danger")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated


def is_market_open():
    """
    Checks if the Indian Stock Market (NSE/BSE) is currently open.
    Hours: 09:15 to 15:30 IST, Monday to Friday.
    """
    # 1. Set Timezone to IST
    india_tz = pytz.timezone('Asia/Kolkata')
    now_india = datetime.now(india_tz)
    
    # 2. Check for Weekends (5 = Saturday, 6 = Sunday)
    if now_india.weekday() >= 5:
        return False
        
    # 3. List of 2026 Market Holidays (Add dates as strings YYYY-MM-DD)
    holidays = [
        "2026-01-26", "2026-03-03", "2026-03-26", "2026-03-31", 
        "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-28"
    ]
    if now_india.strftime('%Y-%m-%d') in holidays:
        return False

    # 4. Check for Market Hours (09:15 to 15:30)
    current_time = now_india.time()
    market_open = datetime.strptime("09:15:00", "%H:%M:%S").time()
    market_close = datetime.strptime("15:30:00", "%H:%M:%S").time()
    
    if market_open <= current_time <= market_close:
        return True
        
    return False


def next_business_day(date: pd.Timestamp) -> pd.Timestamp:
    """Return the next Mon–Fri day after the given date."""
    date += pd.Timedelta(days=1)
    while date.weekday() >= 5:
        date += pd.Timedelta(days=1)
    return date


def build_forecast_dates(last_date: pd.Timestamp, n: int = 5) -> list:
    """Return n business-day date strings starting after last_date."""
    dates, current = [], last_date
    for _ in range(n):
        current = next_business_day(current)
        dates.append(current.strftime('%b %d'))
    return dates


def compute_signal_strength(forecast: list) -> list:
    """
    Score each forecast day by stability (lower swing = higher score).
    Day 1 baseline is 82. Each subsequent day decays based on volatility.
    """
    scores = [82.0]
    for i in range(1, len(forecast)):
        swing = abs(forecast[i] - forecast[i - 1]) / (forecast[i - 1] + 1e-9) * 100
        scores.append(round(max(40.0, 82.0 - i * 3 - swing * 2), 1))
    return scores


def _csv_response(headers: list, rows, filename: str) -> Response:
    """Build a UTF-8 BOM CSV download response."""
    si = StringIO()
    si.write('\ufeff')          # BOM — makes Excel open it correctly
    cw = csv.writer(si)
    cw.writerow(headers)
    cw.writerows(rows)
    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))


# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for('admin') if current_user.role == 'ADMIN' else url_for('dashboard'))
    return redirect(url_for('login'))


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email    = request.form.get('email',    '').strip().lower()
        fullname = request.form.get('fullname', '').strip()
        password = request.form.get('password', '')

        if not email or not fullname or not password:
            flash('All fields are required.', 'warning')
            return render_template('register.html')

        # Email Validation
        if not is_valid_email(email):
            flash('Please enter a valid email address.', 'danger')
            return render_template('register.html')

        # Password Constraint Check
        is_valid, msg = validate_password_strength(password)
        if not is_valid:
            flash(msg, 'danger')
            return render_template('register.html')

        hashed_pw   = bcrypt.generate_password_hash(password).decode('utf-8')
        admin_email = os.getenv('ADMIN_EMAIL', '').strip().lower()
        role        = 'ADMIN' if email == admin_email else 'USER'

        try:
            db.session.add(User(full_name=fullname, email=email,
                                password=hashed_pw, role=role))
            db.session.commit()
            flash('Account created! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            print(f"[Registration Error] {e}")
            flash('Email already exists or registration failed.', 'danger')

    return render_template('register.html')


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('admin') if current_user.role == 'ADMIN' else url_for('dashboard'))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            
            # --- THE FIX: Define lifetime BEFORE setting session.permanent ---
            if remember:
                app.permanent_session_lifetime = timedelta(days=7)
                session.permanent = True
            else:
                # This ensures the browser shows a Date/Time, not just "Session"
                app.permanent_session_lifetime = timedelta(hours=1)
                session.permanent = True

            # login_user tells Flask-Login whether to send the 'remember_token' cookie
            login_user(user, remember=remember)
            
            log_action_for(user.email, "User Logged In")
            flash(f'Welcome back, {user.full_name}!', 'success')

            next_page = request.args.get('next')
            return redirect(next_page or (
                url_for('admin') if user.role == 'ADMIN' else url_for('dashboard')
            ))

        flash('Login unsuccessful. Please check your email and password.', 'danger')

    return render_template('login.html')


@app.route("/logout")
def logout():
    """
    No @login_required — intentional.
    logout_user() is a safe no-op on a stale/expired session.
    Clicking Logout on an expired tab just cleanly lands on /login.
    """
    if current_user.is_authenticated:
        log_action("User Logged Out")

    logout_user()
    session.clear()

    # Explicitly expire both cookies so the browser discards them immediately
    response = redirect(url_for('login'))
    response.set_cookie(
        app.config.get('SESSION_COOKIE_NAME', 'session'), '', expires=0
    )
    response.set_cookie('remember_token', '', expires=0)

    flash("You have been logged out.", "info")
    return response


# ─────────────────────────────────────────────
# USER ROUTES
# ─────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    symbol   = request.args.get('symbol', 'TSLA').upper()
    news     = get_market_news()
    history  = get_stock_history(symbol)
    trending = get_trending_stocks()
    indices  = get_market_indices()
    market_open_bool = is_market_open()

    recent_preds = (
        Prediction.query
        .filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(5).all()
    )

    return render_template(
        'dashboard.html',
        user=current_user,
        news=news,
        history=history,
        recent_preds=recent_preds,
        symbol=symbol,
        trending=trending,
        is_market_open=market_open_bool,
        market_status="Live" if is_market_open() else "Closed",
        indices=indices,
    )


@app.route("/api/trending")
@login_required
def api_trending():
    return {"success": True, "stocks": get_trending_stocks(), "is_market_open": is_market_open()}


@app.route("/predict", methods=['POST'])
@login_required
def predict():
    symbol      = request.form.get('symbol', '').upper().strip()
    model_path  = f'models/{symbol}_model.keras'
    scaler_path = f'models/{symbol}_scaler.pkl'

    # ── 1. Download ───────────────────────────────────────────
    try:
        df = yf.download(
            symbol, period="120d", progress=False,
            multi_level_index=False, auto_adjust=True,
        )
    except Exception as e:
        flash(f"Failed to download data for {symbol}: {e}", "danger")
        return redirect(url_for('dashboard'))

    if df.empty or len(df) < WINDOW_SIZE + 50:
        flash(f"Not enough market data for '{symbol}'.", "danger")
        return redirect(url_for('dashboard'))

    # ── 2. Clean ──────────────────────────────────────────────
    df = df[['Close']].copy()
    df.dropna(inplace=True)
    df = df[df['Close'] > 0]

    if len(df) < WINDOW_SIZE + 5:
        flash(f"Insufficient clean data for {symbol}.", "danger")
        return redirect(url_for('dashboard'))

    # ── 3. Extract prices ─────────────────────────────────────
    try:
        current_price = float(df['Close'].iloc[-1])
        prev_close    = float(df['Close'].iloc[-2])
        if np.isnan(current_price) or np.isnan(prev_close) or current_price <= 0:
            raise ValueError("Invalid price values.")
    except Exception:
        flash(f"Price data for {symbol} is unstable. Try again later.", "danger")
        return redirect(url_for('dashboard'))

    quote_data = {
        'current':    round(current_price, 2),
        'prev_close': round(prev_close, 2),
        'change':     round(current_price - prev_close, 2),
        'percent':    round(((current_price - prev_close) / prev_close) * 100, 2),
    }

    # ── 4. Initialise (prevents UnboundLocalError) ────────────
    forecast_5day = []
    method_used   = None

    # ── 4A. LSTM ──────────────────────────────────────────────
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            scaler        = joblib.load(scaler_path)
            model         = load_model(model_path)
            scaled        = scaler.transform(df[['Close']].values)
            current_batch = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)

            forecast_scaled = []
            for _ in range(5):
                pred = model.predict(current_batch, verbose=0)
                forecast_scaled.append(float(pred[0, 0]))
                current_batch = np.append(
                    current_batch[:, 1:, :], pred.reshape(1, 1, 1), axis=1
                )

            forecast_5day = [
                round(float(v), 2) for v in
                scaler.inverse_transform(
                    np.array(forecast_scaled).reshape(-1, 1)
                ).flatten()
            ]
            method_used = "LSTM Deep Learning"

        except Exception as e:
            print(f"[LSTM Error — {symbol}] {e}")
            method_used = None

    # ── 4B. Technical Analysis Fallback ──────────────────────
    if method_used is None:
        is_bullish = current_price > float(
            df['Close'].rolling(window=20).mean().iloc[-1]
        )
        temp = current_price
        forecast_5day = []
        for _ in range(5):
            temp = temp * (1 + (0.004 if is_bullish else -0.004)
                           + random.uniform(-0.002, 0.002))
            forecast_5day.append(round(temp, 2))
        method_used = "Technical Analysis (SMA + Volatility)"

    # ── 5. Scores & recommendation ────────────────────────────
    confidences     = compute_signal_strength(forecast_5day)
    predicted_price = forecast_5day[0]
    change_pct      = ((predicted_price - current_price) / current_price) * 100
    rec = "BUY" if change_pct > 1.5 else "SELL" if change_pct < -1.5 else "HOLD"

    # ── 6. Forecast dates ─────────────────────────────────────
    forecast_data = list(zip(
        build_forecast_dates(df.index[-1]), forecast_5day, confidences
    ))

    # ── 7. Save to DB ─────────────────────────────────────────
    if not np.isnan(predicted_price):
        try:
            db.session.add(Prediction(
                symbol=symbol,
                price=round(float(predicted_price), 2),
                recommendation=rec,
                user_id=current_user.id,
            ))
            db.session.commit()
            log_action(f"Prediction saved for {symbol}")
        except Exception as e:
            db.session.rollback()
            print(f"[DB Save Error] {e}")

    # ── 8. History for chart ──────────────────────────────────
    history_raw = get_stock_history(symbol)
    history = {
        'labels': history_raw.get('labels', []),
        'prices': history_raw.get('values', history_raw.get('prices', [])),
    }

    return render_template(
        'prediction_result.html',
        symbol=symbol,
        current=round(current_price, 2),
        predicted=round(predicted_price, 2),
        rec=rec,
        quote=quote_data,
        forecast_data=forecast_data,
        method=method_used,
        history=history,
    )


@app.route("/compare", methods=['GET', 'POST'])
@login_required
def compare():
    if request.method == 'POST':
        s1 = request.form.get('symbol1', '').upper().strip()
        s2 = request.form.get('symbol2', '').upper().strip()
        data = get_comparison_data(s1, s2)
        if not data.get('success'):
            flash(data.get('error', 'Comparison failed.'), 'danger')
            return redirect(url_for('dashboard'))
        return render_template('compare.html', data=data)
    return render_template('compare_search.html')


@app.route("/stocks")
@login_required
def stocks_list():
    trained_stocks = [
        {'symbol': 'TCS.NS',        'name': 'Tata Consultancy Services',     'trained': True},
        {'symbol': 'HDFCBANK.NS',   'name': 'HDFC Bank Ltd',                 'trained': True},
        {'symbol': 'RELIANCE.NS',   'name': 'Reliance Industries',           'trained': True},
        {'symbol': 'INFY.NS',       'name': 'Infosys Ltd',                   'trained': True},
        {'symbol': 'ITC.NS',        'name': 'ITC Ltd',                       'trained': True},
        {'symbol': 'SUNPHARMA.NS',  'name': 'Sun Pharmaceutical Industries', 'trained': True},
        {'symbol': 'TSLA',          'name': 'Tesla Inc.',                    'trained': True},
        {'symbol': 'AAPL',          'name': 'Apple Inc.',                    'trained': True},
        {'symbol': 'ADANIENT.NS',   'name': 'Adani Enterprises Ltd',         'trained': True},
        {'symbol': 'BAJFINANCE.NS', 'name': 'Bajaj Finance Ltd',             'trained': True},
    ]
    other_stocks = [
        {'symbol': 'SBIN.NS',       'name': 'State Bank of India',           'trained': False},
        {'symbol': 'BHARTIARTL.NS', 'name': 'Bharti Airtel Ltd',             'trained': False},
        {'symbol': 'HINDUNILVR.NS', 'name': 'Hindustan Unilever Ltd',        'trained': False},
        {'symbol': 'LT.NS',         'name': 'Larsen & Toubro Ltd',           'trained': False},
        {'symbol': 'AXISBANK.NS',   'name': 'Axis Bank Ltd',                 'trained': False},
        {'symbol': 'KOTAKBANK.NS',  'name': 'Kotak Mahindra Bank',           'trained': False},
        {'symbol': 'MARUTI.NS',     'name': 'Maruti Suzuki India Ltd',       'trained': False},
        {'symbol': 'TITAN.NS',      'name': 'Titan Company Ltd',             'trained': False},
        {'symbol': 'GOOGL',         'name': 'Alphabet Inc.',                 'trained': False},
        {'symbol': 'AMZN',          'name': 'Amazon.com Inc.',               'trained': False},
    ]
    return render_template('stocks.html', stocks=trained_stocks + other_stocks)


@app.route("/profile", methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        user           = db.session.get(User, current_user.id)
        email          = request.form.get('email', '').strip().lower()
        
        # NEW: Email Validation on update
        if not is_valid_email(email):
            flash('Invalid email format.', 'danger')
            return redirect(url_for('profile'))

        user.full_name = request.form.get('fullname', '').strip()
        user.email     = email

        new_pw = request.form.get('password', '')
        if new_pw:
            # NEW: Password Constraint Check on update
            is_valid, msg = validate_password_strength(new_pw)
            if not is_valid:
                flash(msg, 'danger')
                return redirect(url_for('profile'))
            user.password = bcrypt.generate_password_hash(new_pw).decode('utf-8')

        try:
            db.session.commit()
            log_action("User updated their profile")
            flash("Profile updated successfully!", "success")
        except Exception:
            db.session.rollback()
            flash("Error: Email might already be in use.", "danger")

        return redirect(url_for('profile'))

    return render_template('profile.html', user=current_user)


# ─────────────────────────────────────────────
# ADMIN ROUTES
# ─────────────────────────────────────────────
@app.route("/admin")
@login_required
@admin_required
def admin():
    stats = {
        "total_users": User.query.count(),
        "total_preds": Prediction.query.count(),
    }
    top_stocks = (
        db.session.query(
            Prediction.symbol,
            func.count(Prediction.symbol).label('count')
        )
        .group_by(Prediction.symbol)
        .order_by(func.count(Prediction.symbol).desc())
        .limit(5).all()
    )

    today = datetime.now()
    chart_labels, chart_values = [], []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        chart_labels.append(d.strftime('%b %d'))
        chart_values.append(
            Prediction.query.filter(
                func.date(Prediction.created_at) == d.strftime('%Y-%m-%d')
            ).count()
        )

    return render_template(
        'admin.html',
        users=User.query.all(),
        stats=stats,
        top_stocks=top_stocks,
        chart_labels=chart_labels,
        chart_values=chart_values,
    )


@app.route("/admin/users")
@login_required
@admin_required
def user_management():
    return render_template('user_management.html', users=User.query.all())


@app.route("/admin/add_user", methods=['POST'])
@login_required
@admin_required
def add_user():
    email     = request.form.get('email',    '').strip().lower()
    full_name = request.form.get('fullname', '').strip()
    password  = request.form.get('password', '')
    role      = request.form.get('role', 'USER')

    # NEW: Constraints for Admin-created users
    if not is_valid_email(email):
        flash('Invalid email format.', 'danger')
        return redirect(url_for('user_management'))

    is_valid, msg = validate_password_strength(password)
    if not is_valid:
        flash(f"User not added. {msg}", 'danger')
        return redirect(url_for('user_management'))

    try:
        db.session.add(User(
            full_name=full_name, email=email,
            password=bcrypt.generate_password_hash(password).decode('utf-8'),
            role=role,
        ))
        db.session.commit()
        log_action(f"Admin created new user: {email}")
        flash('User added successfully!', 'success')
    except Exception:
        db.session.rollback()
        flash('Error: Email already exists.', 'danger')

    return redirect(url_for('user_management'))


@app.route("/admin/edit_user/<int:user_id>", methods=['POST'])
@login_required
@admin_required
def edit_user(user_id):
    user           = db.get_or_404(User, user_id)
    user.full_name = request.form.get('fullname', '').strip()
    user.email     = request.form.get('email',    '').strip().lower()
    user.role      = request.form.get('role', 'USER')

    new_pw = request.form.get('password', '')
    if new_pw:
        user.password = bcrypt.generate_password_hash(new_pw).decode('utf-8')

    try:
        db.session.commit()
        log_action(f"Admin updated user ID: {user_id}")
        flash('User updated successfully!', 'success')
    except Exception:
        db.session.rollback()
        flash('Error updating user.', 'danger')

    return redirect(url_for('user_management'))


@app.route("/admin/delete_user/<int:user_id>", methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    user = db.get_or_404(User, user_id)

    if user.id == current_user.id:
        flash("You cannot delete your own admin account!", "danger")
        return redirect(url_for('user_management'))

    try:
        db.session.delete(user)
        db.session.commit()
        log_action(f"Admin deleted user ID: {user_id}")
        flash(f"User {user.full_name} deleted successfully!", "success")
    except Exception as e:
        db.session.rollback()
        print(f"[Delete Error] {e}")
        flash("Error deleting user.", "danger")

    return redirect(url_for('user_management'))


@app.route("/admin/export_users")
@login_required
@admin_required
def export_users():
    log_action("Admin exported user list to CSV")
    users = User.query.all()
    return _csv_response(
        ['ID', 'Full Name', 'Email', 'Role', 'Joined Date'],
        [[u.id, u.full_name, u.email, u.role,
          u.created_at.strftime('%Y-%m-%d')] for u in users],
        "user_list.csv",
    )


@app.route("/admin/logs")
@login_required
@admin_required
def view_logs():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).all()
    return render_template('audit_logs.html', logs=logs)


@app.route("/admin/export_logs")
@login_required
@admin_required
def export_logs():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).all()
    log_action("Admin exported system logs to CSV")
    return _csv_response(
        ['Log ID', 'User Email', 'Action Performed', 'Timestamp'],
        [[l.id, l.user_email, l.action,
          l.timestamp.strftime('%Y-%m-%d %H:%M:%S')] for l in logs],
        "system_audit_logs.csv",
    )


@app.route("/admin/analytics")
@login_required
@admin_required
def admin_analytics():
    sys_info = {
        'cpu_usage':      psutil.cpu_percent(interval=0.1),
        'ram_usage':      psutil.virtual_memory().percent,
        'disk_free':      round(psutil.disk_usage('/').free / (1024 ** 3), 2),
        'python_version': os.sys.version.split()[0],
    }
    plot_folder   = 'static/plots'
    trained_plots = (
        sorted(f for f in os.listdir(plot_folder) if f.endswith('.png'))
        if os.path.exists(plot_folder) else []
    )
    # Pass training status so the template can show/hide the Start button
    training_active = is_training_running()

    return render_template(
        'admin_analytics.html',
        sys_info=sys_info,
        trained_plots=trained_plots,
        training_active=training_active,
    )


@app.route("/admin/run_training", methods=['POST'])
@login_required
@admin_required
def run_training():
    """
    Starts the training script in a background subprocess.
    LOCK FILE PATTERN — prevents multiple simultaneous training runs.
    """
    if is_training_running():
        flash(
            "Training is already running. Please wait for it to finish "
            "before starting another run.",
            "warning"
        )
        return redirect(url_for('admin_analytics'))

    try:
        with open("training_log.txt", "w") as lf:
            proc = subprocess.Popen(
    [sys.executable, 'train_top_10.py'],
    stdout=lf,
    stderr=lf,
    cwd=os.path.dirname(os.path.abspath(__file__))
)

        # Write the child process PID into the lock file
        with open(TRAINING_LOCK, 'w') as lock_file:
            lock_file.write(str(proc.pid))

        log_action(f"Admin started model training (PID: {proc.pid})")
        flash(
            f"Training started (PID {proc.pid}). "
            "Check training_log.txt for live progress.",
            "success"
        )

    except Exception as e:
        _remove_lock()   # Clean up if Popen succeeded but write failed
        flash(f"Failed to start training: {e}", "danger")

    return redirect(url_for('admin_analytics'))


@app.route("/admin/training_status")
@login_required
@admin_required
def training_status():
    """
    Lightweight JSON endpoint polled by the analytics page
    to update the training status badge without a full page reload.
    """
    running = is_training_running()
    log_tail = []

    # Read last 20 lines of training_log.txt for live progress display
    if os.path.exists("training_log.txt"):
        try:
            with open("training_log.txt", "r") as f:
                log_tail = f.readlines()[-20:]
        except IOError:
            pass

    return {
        "running": running,
        "log": "".join(log_tail),
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Clean up any stale lock from a previous crashed run
    if not is_training_running():
        _remove_lock()

    with app.app_context():
        db.create_all()
    app.run(debug=True)