## StockSense AI - An AI Based Stock Price Prediction System

## Overview

StockSense AI bridges the gap between traditional technical analysis and modern Deep Learning. By utilizing Long Short-Term Memory (LSTM) networks, the system identifies non-linear patterns in historical stock data to provide actionable financial recommendations.

## Key Features

### AI & Analytics

- **LSTM Deep Learning:** High-accuracy 5-day price forecasting for top-tier stocks.
- **Technical Fallback:** Hybrid prediction engine using Simple Moving Averages (SMA) and Volatility modeling for non-trained tickers.
- **Signal Strength Scoring:** Mathematical confidence metric based on forecast stability.
- **Interactive Visuals:** Real-time historical price charts and training convergence (Loss) plots.

### User Experience

- **Personalized Dashboard:** Track recent prediction history and global market indices (NIFTY 50, SENSEX).
- **Profile Management:** Securely update personal details and encrypted passwords.
- **Stock Comparison:** Side-by-side performance analysis of two different symbols.

### Admin & Security

- **System Telemetry:** Real-time monitoring of CPU, RAM, and Storage health during AI training.
- **Training Safety:** Persistent Lock-File mechanism and PID tracking to prevent hardware overload.
- **Audit Logs:** Immutable tracking of all administrative actions with **CSV Export** capability.
- **User Management:** Full CRUD system with secure Bcrypt password hashing.

## Tech Stack

- **Backend:** Python (Flask),Flask-Login, Flask-Bcrypt
- **Database:** MySQL (via XAMPP/SQLAlchemy)
- **Frontend:** HTML5, Tailwind CSS, Jinja2, Chart.js
- **Machine Learning:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy
- **Data APIs:** Yahoo Finance (yfinance), Finnhub API

## Installation & Setup

**Global Environment Warning:** This project is configured to run in a global Python environment. To ensure the AI models and session management function correctly, please ensure no conflicting versions of TensorFlow (2.15+) or Flask (3.0+) are currently active. If errors occur, it is recommended to run pip uninstall on existing versions before installing the project requirements.

### 1. Prerequisites

- Python 3.9+
- XAMPP (For MySQL)
- A stable internet connection (for real-time API data)
- **Note:** Ensure no conflicting versions of TensorFlow or Flask are installed globally.

### 2. Environment Configuration

Create a `.env` file in the root directory:

```text
FLASK_SECRET=your_dev_key_123
FINNHUB_API_KEY=your_api_key_here
ADMIN_EMAIL=your_admin_email@example.com
```

### 3. Database Setup

Start Apache and MySQL in XAMPP.
Go to http://localhost/phpmyadmin.
Create a new database named stock_db.

    The application uses SQLAlchemy to automatically generate tables on first run.

### 4. Install dependencies

pip install -r requirements.txt

### 5. AI Model Training : To generate the initial LSTM models for the top 10 stocks:

python train_top_10.py

### 6. Run the application

python app.py

### Project Structure

├── app.py # Flask Server, Auth, & Analytics Routes
├── train_top_10.py # Background LSTM Training Pipeline
├── training.lock # Runtime Lock-file (auto-generated)
├── services/ # Market API Logic & Data Normalization
├── models/ # Saved .keras models and .pkl scalers
├── templates/ # Jinja2 HTML Templates (MVC View)
├── static/  
│ ├── plots/ # Training Metrics & Loss Charts
│ └── assets/ # CSS/JS and Chart.js logic
└── requirements.txt # Project Dependencies

This project implements Resource-Aware Machine Learning. The training pipeline includes automatic RAM gatekeeping and CPU thread limiting to ensure system stability during deep learning operations. All security protocols (Bcrypt, HttpOnly Cookies, CSRF protection) follow modern industry standards.
