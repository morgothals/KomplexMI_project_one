import os
import pandas as pd

# ----------------------------------
# CONFIG
# ----------------------------------
DATA_DIR = r"C:\Users\gaspa\Downloads\archive"  # <-- ÁLLÍTSD ÁT
OUTPUT_FILE = "market_datas.csv"

# ----------------------------------
# Helper: format ISO datetime
# ----------------------------------
def to_iso(date_str):
    return pd.to_datetime(date_str).strftime("%Y-%m-%dT00:00:00Z")

# ----------------------------------
# Process all CSV files
# ----------------------------------
frames = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".csv"):
        filepath = os.path.join(DATA_DIR, filename)
        symbol = filename.replace(".csv", "").upper() + "-USD"

        print(f"Processing: {symbol}")

        df = pd.read_csv(filepath)

        # Convert date
        df["datetime"] = df["Date"].apply(to_iso)

        # Add symbol
        df["symbol"] = symbol

        # Keep only needed columns (Volume valódi érték)
        df2 = df[["datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]].copy()
        df2.columns = ["datetime", "symbol", "open", "high", "low", "close", "volume"]

        # Töröljük a felesleges COIN_ előtagot
        df2["symbol"] = df2["symbol"].str.replace("COIN_", "", regex=False)

        frames.append(df2)

# Combine all
full_df = pd.concat(frames, ignore_index=True)

# Sort by datetime
full_df = full_df.sort_values(by="datetime")

# Save
full_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nDONE! Saved full dataset to: {OUTPUT_FILE}")
