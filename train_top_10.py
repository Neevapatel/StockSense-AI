import numpy as np
import pandas as pd
import yfinance as yf
import os
import sys
import joblib
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# ─────────────────────────────────────────────
# LAPTOP SAFETY — GPU / CPU / MEMORY LIMITS
# ─────────────────────────────────────────────

# 1. Suppress verbose TensorFlow logs (only show errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 2. Stop TensorFlow from grabbing ALL GPU memory at once.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass   # Must be set before GPUs are initialised — safe to ignore if late

# 3. Limit TensorFlow to use at most 4 CPU threads.
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
WINDOW_SIZE   = 60      # Days of history fed into LSTM
EPOCHS        = 50      # Max epochs — EarlyStopping will cut short
BATCH_SIZE    = 32
TRAIN_SPLIT   = 0.80    # 80% train, 20% test
PATIENCE      = 7       # EarlyStopping patience
TRAINING_LOCK = "training.lock"

# RAM safety gate: if free RAM drops below this (MB), skip remaining stocks
MIN_FREE_RAM_MB = 500

STOCKS = {
    'TCS.NS':       'IT Sector',
    'HDFCBANK.NS':  'Banking',
    'RELIANCE.NS':  'Energy',
    'INFY.NS':      'IT Services',
    'ITC.NS':       'FMCG',
    'SUNPHARMA.NS': 'Pharma',
    'TSLA':         'US Tech',
    'AAPL':         'US Blue-chip',
    'ADANIENT.NS':  'Metals/Energy',
    'BAJFINANCE.NS':'Finance',
}


# ─────────────────────────────────────────────
# LOCK FILE HELPERS
# ─────────────────────────────────────────────
def remove_lock():
    """Delete the training lock file so app.py knows we're done."""
    try:
        os.remove(TRAINING_LOCK)
        print("  [LOCK] Training lock released.")
    except FileNotFoundError:
        pass


# ─────────────────────────────────────────────
# MODEL / PLOT HELPERS
# ─────────────────────────────────────────────
def create_sequences(data: np.ndarray, window: int):
    """Slide a window over 1-D scaled data → (X, y) arrays."""
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_model(window: int) -> Sequential:
    """Two-layer stacked LSTM with dropout."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def save_loss_plot(history, symbol: str):
    """Save training + validation loss curves."""
    plt.figure(figsize=(7, 4))
    plt.plot(history.history['loss'],     label='Train Loss', color='indigo')
    plt.plot(history.history['val_loss'], label='Val Loss',   color='orange')
    plt.title(f'Training Loss — {symbol}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'static/plots/{symbol}_loss.png', dpi=100)
    plt.close()   # ← CRITICAL: always close to free memory


def save_prediction_plot(actual: np.ndarray, predicted: np.ndarray, symbol: str):
    """Save actual vs predicted price chart on the test set."""
    plt.figure(figsize=(10, 4))
    plt.plot(actual,    label='Actual Price',    color='steelblue')
    plt.plot(predicted, label='Predicted Price', color='tomato', linestyle='--')
    plt.title(f'Actual vs Predicted — {symbol}')
    plt.xlabel('Test Days')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'static/plots/{symbol}_pred.png', dpi=100)
    plt.close()   # ← CRITICAL: always close to free memory


def check_ram(symbol: str) -> bool:
    """
    Return False and print a warning if free RAM is dangerously low.
    Prevents the laptop from running out of memory mid-training.
    """
    free_mb = psutil.virtual_memory().available / (1024 ** 2)
    if free_mb < MIN_FREE_RAM_MB:
        print(f"  [SKIP] {symbol}: only {free_mb:.0f} MB RAM free "
              f"(threshold: {MIN_FREE_RAM_MB} MB). Skipping to protect system.")
        return False
    return True


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────
def train_diverse_stocks():
    os.makedirs('models',       exist_ok=True)
    os.makedirs('static/plots', exist_ok=True)

    results = []

    for symbol, sector in STOCKS.items():
        print(f"\n{'='*55}")
        print(f"  Training : {symbol}  ({sector})")
        print(f"{'='*55}")

        # ── RAM safety check before each stock ───────────────
        if not check_ram(symbol):
            continue

        # ── 1. Download ───────────────────────────────────────
        try:
            df = yf.download(
                symbol,
                period="5y",
                progress=False,
                multi_level_index=False,
                auto_adjust=True,
            )
        except Exception as e:
            print(f"  [SKIP] Download failed for {symbol}: {e}")
            continue

        if df.empty or len(df) < WINDOW_SIZE + 50:
            print(f"  [SKIP] Not enough data for {symbol} ({len(df)} rows).")
            continue

        # ── 2. Clean ──────────────────────────────────────────
        df = df[['Close']].copy()
        df.dropna(inplace=True)
        df = df[df['Close'] > 0]
        data = df.values.astype(float)

        # ── 3. Scale ──────────────────────────────────────────
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data)
        joblib.dump(scaler, f'models/{symbol}_scaler.pkl')

        # ── 4. Train / Test Split ─────────────────────────────
        split_idx  = int(len(scaled) * TRAIN_SPLIT)
        train_data = scaled[:split_idx]
        test_data  = scaled[split_idx - WINDOW_SIZE:]   # lookback overlap

        if len(train_data) <= WINDOW_SIZE:
            print(f"  [SKIP] Train data too small for {symbol}.")
            continue

        X_train, y_train = create_sequences(train_data, WINDOW_SIZE)
        X_test,  y_test  = create_sequences(test_data,  WINDOW_SIZE)
        X_train = X_train.reshape(-1, WINDOW_SIZE, 1)
        X_test  = X_test.reshape(-1,  WINDOW_SIZE, 1)

        # ── 5. Build + Train ──────────────────────────────────
        model = build_model(WINDOW_SIZE)

        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=PATIENCE,
                    restore_best_weights=True,
                    verbose=1,
                )
            ],
            verbose=1,
        )

        # ── 6. Evaluate ───────────────────────────────────────
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred_actual = scaler.inverse_transform(y_pred_scaled)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = float(np.sqrt(np.mean((y_pred_actual - y_test_actual) ** 2)))
        print(f"  Test RMSE: {rmse:.4f}")
        results.append({'symbol': symbol, 'sector': sector, 'rmse': round(rmse, 4)})

        # ── 7. Save model ─────────────────────────────────────
        model.save(f'models/{symbol}_model.keras')

        # ── 8. Save plots ─────────────────────────────────────
        save_loss_plot(history, symbol)
        save_prediction_plot(y_test_actual, y_pred_actual, symbol)

        # ── 9. Explicit memory cleanup ────────────────────────
        # TensorFlow holds onto the model graph in memory.
        # Deleting it and clearing the Keras session between stocks
        # keeps RAM usage flat across all 10 training runs.
        del model
        tf.keras.backend.clear_session()

        print(f"  [DONE] {symbol}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  TRAINING COMPLETE — RMSE Summary")
    print(f"{'='*55}")
    for r in results:
        print(f"  {r['symbol']:<18} ({r['sector']:<14})  RMSE = {r['rmse']}")

    if not results:
        print("  No models were trained successfully.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        train_diverse_stocks()
    finally:
        # ── Always release the lock, even if training crashes ─
        # The 'finally' block runs whether training succeeded,
        # failed with an exception, or was killed mid-way.
        # This guarantees the lock file is ALWAYS cleaned up,
        # so we can always start a new run afterwards.
        remove_lock()
        print("  Training process exited.")
        sys.exit(0)