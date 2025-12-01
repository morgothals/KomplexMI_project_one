# main.py
import argparse
import pandas as pd

from modules.config import (
    MARKET_DATA_CSV,
    MARKET_FEATURES_CSV,
    ALL_FEATURES_CSV,  # ha már nem használod, ezt akár el is hagyhatjuk
)
from modules.data_collector import (
    update_market_data_csv,
    update_onchain_data,
    update_macro_data,
    update_intraday_minute_data,
)
from modules.feature_engineering import add_all_features
from modules.feature_assembler import build_all_features
from modules.sentiment_analyzer import build_sentiment_timeseries
from modules.advisor import generate_advice
from modules.forecast_model import train_model
from modules.longterm_features import build_longterm_btc_features
from modules.longterm_forecaster import run_build_long_horizon_curve
from modules.log_curve_forecaster import run_log_regression_curve

def cmd_update_data():
    print(">>> Binance OHLCV frissítés...")
    df_mkt = update_market_data_csv()
    print(f"Market data shape: {df_mkt.shape}")

    print(">>> On-chain adatok frissítése (Blockchair + Blockchain.com)...")
    df_onchain = update_onchain_data()
    print(f"On-chain shape: {df_onchain.shape}")

    print(">>> Makró adatok frissítése (Yahoo Finance)...")
    df_macro = update_macro_data()
    print(f"Makró shape: {df_macro.shape}")

    print(">>> Hír + sentiment idősor build (CoinDesk, Reddit, Cointelegraph + Fear&Greed)...")
    df_sent = build_sentiment_timeseries()
    print(f"Sentiment shape: {df_sent.shape}")
    
    print(">>> Intraday 1m OHLCV frissítés (mai nap)...")
    df_1m = update_intraday_minute_data()
    print(f"Intraday 1m shape: {df_1m.shape}")

    print(">>> Hosszútávú BTC feature dataset (15 napos) építése...")
    df_long = build_longterm_btc_features()
    print(f"Hosszútávú feature shape: {df_long.shape}")

    print("Kész.")


def cmd_build_features():
    print(">>> Feature engineering (technikai indikátorok)...")
    df_mkt = pd.read_csv(MARKET_DATA_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df_fe = add_all_features(df_mkt)
    df_fe.to_csv(MARKET_FEATURES_CSV, index_label="timestamp")
    print(f"Market features shape: {df_fe.shape}")


def cmd_build_all_features():
    print(">>> Összes feature (market + on-chain + macro + sentiment) összeállítása...")
    df_all = build_all_features(resample_rule="1H")
    print(f"All features shape: {df_all.shape}")


def cmd_train(epochs=10):
    print(">>> Model tanítása (training_features_1h alapján)...")
    train_model(epochs=epochs)


def cmd_advise():
    print(">>> Tanács generálása...")
    advice = generate_advice()
    print("Jelzés:", advice["signal"])
    print("Utolsó záróár:", advice["last_close"])
    print("Következő ár előrejelzés:", advice["next_price_pred"])
    print("Relatív változás (pred):", advice["rel_change_pred"])
    print("Fear & Greed:", advice["fear_greed"])
    print("News sentiment:", advice["news_sentiment"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "update_data",
        "build_features",
        "build_all_features",
        "train",
        "advise",
        "build_long_curve", 
        "log_curve",
    ])
    parser.add_argument("--epochs", type=int, default=10)  # most nem használjuk, de maradhat
    args = parser.parse_args()

    if args.command == "update_data":
        cmd_update_data()
    elif args.command == "build_features":
        cmd_build_features()
    elif args.command == "build_all_features":
        cmd_build_all_features()
    elif args.command == "train":
        cmd_train(epochs=args.epochs)
    elif args.command == "advise":
        cmd_advise()
    elif args.command == "build_long_curve":
        # tetszés szerint módosíthatod az éveket
        run_build_long_horizon_curve(start_year=2012, end_year=2031, sigma_multiplier=1.0)
    elif args.command == "log_curve":
        run_log_regression_curve(end_year=2030, sigma_mult=1.0)