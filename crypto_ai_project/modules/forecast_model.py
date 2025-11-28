# modules/forecast_model.py

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from .config import (
    TRAINING_FEATURES_CSV,
    FORECAST_MODEL_PATH,
    FORECAST_SCALER_PATH,
    LOOKBACK,
)


def load_training_data():
    """
    Teljes training_features_1h.csv betöltése.

    Target: log-return a close árra:
        log_return_t = ln(close_t / close_{t-1})

    Features: minden oszlop (beleértve a close-t is), kivéve a log_return (target).
    """
    df = pd.read_csv(TRAINING_FEATURES_CSV, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    if "close" not in df.columns:
        raise RuntimeError("TRAINING_FEATURES_CSV nem tartalmaz 'close' oszlopot.")

    # log-return kiszámítása
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # az első sor NaN (mert nincs előző ár) -> dobjuk
    df = df.dropna(subset=["log_return"])

    # target: log_return (N x 1)
    y = df["log_return"].values.reshape(-1, 1)

    # features: minden más (close-t is benne hagyjuk feature-ként) closet is kiveszem inkább
    X = df.drop(columns=["log_return", "close"]).values

    return df, X, y



def build_sequences(X, y, lookback=LOOKBACK):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)


def train_model(epochs: int = 50, batch_size: int = 32, patience: int = 5):
    """
    LSTM modell betanítása a training_features_1h.csv alapján.

    Target: a következő időlépés (1H) log-return-je a close árra.
    A modell tehát log-return-t jósol, amit utána ár-változásként tudunk visszafejteni.
    """
    df, X_raw, y_raw = load_training_data()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    X_seq, y_seq = build_sequences(X_scaled, y_scaled, lookback=LOOKBACK)

    if len(X_seq) < 10:
        raise RuntimeError("Túl kevés adat a tanításhoz. Ellenőrizd a training_features_1h.csv méretét.")

    split = int(len(X_seq) * 0.9)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(LOOKBACK, X_seq.shape[2])),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse")

    early = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    checkpoint_cb = ModelCheckpoint(
        filepath=str(FORECAST_MODEL_PATH),   # ugyanoda menthet, vagy pl. 'models/forecast_best.h5'
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early, checkpoint_cb],
        verbose=1,
    )

    # Modell + skálázók mentése
    model.save(FORECAST_MODEL_PATH)
    joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, FORECAST_SCALER_PATH)

    print("Model trained and saved:", FORECAST_MODEL_PATH)
    print("Scalers saved:", FORECAST_SCALER_PATH)


def load_trained_model():
    """
    Betölti a tanított Keras modellt és a skálázókat.
    """
    model = load_model(FORECAST_MODEL_PATH)
    scalers = joblib.load(FORECAST_SCALER_PATH)
    scaler_X = scalers["scaler_X"]
    scaler_y = scalers["scaler_y"]
    return model, scaler_X, scaler_y


def predict_next_close():
    """
    A training_features_1h.csv utolsó LOOKBACK sorából becsüli a következő close árat.

    A modell valójában a KÖVETKEZŐ 1H LOG-RETURN-t jósolja:
        r_hat = log_return_{t+1}

    Ebből számítjuk ki a következő árat:
        close_{t+1} = close_t * exp(r_hat)

    Visszaadja:
        (predicted_close, last_close, last_row_df)
    """
    df, X_raw, y_raw = load_training_data()
    if len(df) <= LOOKBACK:
        raise RuntimeError("Nincs elég sor a training_features_1h.csv-ben a predikcióhoz.")

    model, scaler_X, scaler_y = load_trained_model()

    # csak az utolsó LOOKBACK sor kell input window-nak
    X_last_window_raw = X_raw[-LOOKBACK:]
    X_last_window_scaled = scaler_X.transform(X_last_window_raw)
    X_input = X_last_window_scaled.reshape(1, LOOKBACK, X_last_window_scaled.shape[1])

    # modell kimenete: skálázott log-return
    y_pred_scaled = model.predict(X_input, verbose=0)

    # skálázás visszaforgatása -> valódi log-return
    pred_log_return = scaler_y.inverse_transform(y_pred_scaled)[0, 0]

    # utolsó valódi close ár (y_raw nem kell hozzá, df-ből vesszük)
    last_close = float(df["close"].iloc[-1])

    # következő ár kiszámítása log-return alapján
    predicted_close = last_close * np.exp(pred_log_return)

    last_row = df.iloc[-1]

    return float(predicted_close), last_close, last_row
