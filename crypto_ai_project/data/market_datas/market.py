import pandas as pd

# Bemeneti és kimeneti fájl
input_file = "btcusd_1-min_data.csv"  # <-- a te CSV fájlod elérési útja
output_file = "market_data.csv"

# CSV beolvasása
df = pd.read_csv(input_file)

# Timestamp Unix idő konvertálása ISO 8601 UTC formátumra
df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s', utc=True)

# Symbol hozzáadása minden sorhoz
df['symbol'] = "COIN_BITCOIN-USD"

# Csak a szükséges oszlopok megtartása és átnevezése
df = df[['datetime','symbol','Open','High','Low','Close','Volume']]
df.columns = ['datetime','symbol','open','high','low','close','volume']

# Mentés CSV-be
df.to_csv(output_file, index=False)

# Ellenőrzés
print(df.head())
