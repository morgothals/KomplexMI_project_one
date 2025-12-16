# main.py
import argparse
import pandas as pd

from modules.config import (
    MARKET_DATA_CSV,
    MARKET_FEATURES_CSV,
    ALL_FEATURES_CSV,  # ha már nem használod, ezt akár el is hagyhatjuk
)


def cmd_update_data():
    from modules.data_collector import (
        update_market_data_csv,
        update_onchain_data,
        update_macro_data,
        update_intraday_minute_data,
    )
    from modules.sentiment_analyzer import build_sentiment_timeseries
    from modules.longterm_features import build_longterm_btc_features

    print(">>> Binance OHLCV frissítés (rövid táv: market_data.csv)...")
    df_mkt = update_market_data_csv()
    print(f"Market data shape: {df_mkt.shape}")

    print(">>> Teljes market history frissítés (binance_market_1h.csv + market_data_full.csv)...")
    try:
        from bootstrap_market_data import build_market_data_full

        df_full = build_market_data_full()
        print(f"Market full shape: {df_full.shape}")
    except Exception as e:
        # Ne álljon meg az egész update, ha a Kaggle hiányzik vagy hálózati hiba van.
        print(f"FIGYELEM: market_data_full frissítés kihagyva/hibás ({e})")

    print(">>> On-chain adatok frissítése (Blockchair + Blockchain.com)...")
    df_onchain = update_onchain_data()
    print(f"On-chain shape: {df_onchain.shape}")

    print(">>> Makró adatok frissítése (Yahoo Finance)...")
    df_macro = update_macro_data()
    print(f"Makró shape: {df_macro.shape}")

    print(">>> Hírek frissítése (news_data_30d.csv)...")
    from modules.sentiment_analyzer import update_news_store
    df_news = update_news_store()
    print(f"News data shape: {df_news.shape}")

    print(">>> Hír + sentiment idősor build (CoinDesk, Reddit, Cointelegraph + Fear&Greed)...")
    df_sent = build_sentiment_timeseries()
    print(f"Sentiment shape: {df_sent.shape}")
    
    print(">>> Intraday 1m OHLCV frissítés (mai nap)...")
    df_1m = update_intraday_minute_data()
    print(f"Intraday 1m shape: {df_1m.shape}")

    print(">>> Market feature store frissítése (market_data_features.csv)...")
    cmd_build_features()

    print(">>> Összes feature store frissítése (all_features.csv)...")
    try:
        cmd_build_all_features()
    except Exception as e:
        print(f"FIGYELEM: all_features build kihagyva/hibás ({e})")

    print(">>> Training feature store frissítése (training_features_1h.csv)...")
    try:
        from build_training_features import build_training_features

        build_training_features()
    except Exception as e:
        print(f"FIGYELEM: training_features_1h build kihagyva/hibás ({e})")

    print(">>> Hosszútávú BTC feature dataset (15 napos) építése...")
    df_long = build_longterm_btc_features()
    print(f"Hosszútávú feature shape: {df_long.shape}")

    print("Kész.")


def cmd_build_features():
    from modules.feature_engineering import add_all_features

    print(">>> Feature engineering (technikai indikátorok)...")
    df_mkt = pd.read_csv(MARKET_DATA_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df_fe = add_all_features(df_mkt)
    df_fe.to_csv(MARKET_FEATURES_CSV, index_label="timestamp")
    print(f"Market features shape: {df_fe.shape}")


def cmd_build_all_features():
    from modules.feature_assembler import build_all_features

    print(">>> Összes feature (market + on-chain + macro + sentiment) összeállítása...")
    df_all = build_all_features(resample_rule="1H")
    print(f"All features shape: {df_all.shape}")


def cmd_train(epochs=10):
    from modules.forecast_model import train_model

    print(">>> Model tanítása (training_features_1h alapján)...")
    train_model(epochs=epochs)


def cmd_advise():
    print(">>> Tanács generálása...")
    from modules.advisor import generate_advice

    advice = generate_advice()

    print("Jelzés:", advice.get("signal"))
    print("Időpont (utolsó sor):", advice.get("timestamp"))
    print("Adat frissesség (óra):", advice.get("data_age_hours"))
    print("Horizont:", advice.get("horizon"))

    print("Utolsó záróár:", advice.get("last_close"))
    print("Következő ár előrejelzés:", advice.get("next_price_pred"))
    print("Predikált változás (%):", advice.get("pred_change_pct"))

    rr = advice.get("recent_returns") or {}
    print("Hozamok: 1h / 24h / 7d (%):", rr.get("ret_1h_pct"), rr.get("ret_24h_pct"), rr.get("ret_7d_pct"))

    mkt = advice.get("market") or {}
    print("Market indikátorok: RSI14 / ATR14 / ret_std_30:", mkt.get("rsi_14"), mkt.get("atr_14"), mkt.get("ret_std_30"))
    print("Trend: MA21 / MA50:", mkt.get("ma_21"), mkt.get("ma_50"))
    print("VWAP / Vol change:", mkt.get("vwap"), mkt.get("vol_change"))

    sent = advice.get("sentiment") or {}
    print("Sentiment: Fear&Greed / NewsSent:", sent.get("fear_greed"), sent.get("news_sentiment"))
    print("Sentiment extra: std / bullish / bearish:", sent.get("news_sentiment_std"), sent.get("bullish_ratio"), sent.get("bearish_ratio"))

    macro = advice.get("macro") or {}
    print("Makró: S&P500 / DXY:", macro.get("sp500_close"), macro.get("dxy_close"))

    on = advice.get("onchain") or {}
    print("On-chain: tx / addr / hash:", on.get("tx_count"), on.get("active_addresses"), on.get("hash_rate"))

    rationale = advice.get("rationale") or []
    if rationale:
        print("Indoklás:", "; ".join([str(x) for x in rationale]))

    notes = advice.get("notes") or []
    if notes:
        print("Megjegyzések:", "; ".join([str(x) for x in notes]))


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
        from modules.longterm_forecaster import run_build_long_horizon_curve
        run_build_long_horizon_curve(start_year=2012, end_year=2031, sigma_multiplier=1.0)
    elif args.command == "log_curve":
        from modules.log_curve_forecaster import run_log_regression_curve
        run_log_regression_curve(end_year=2030, sigma_mult=1.0)