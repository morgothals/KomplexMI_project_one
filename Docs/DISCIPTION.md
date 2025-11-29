
---

## 🔷 Projekt összefoglaló – `crypto_ai_project`

Ez egy Python alapú **crypto befektetési tanácsadó rendszer** Bitcoinra fókuszálva.
Fő funkciók:

* Piaci adatok gyűjtése (Binance OHLCV + intraday 1m).
* On-chain adatok (Blockchain.com charts API).
* Makró adatok (S&P500, DXY – Yahoo Finance).
* Hír- és sentiment elemzés (CoinDesk, Reddit, Cointelegraph + Fear & Greed index).
* Feature engineering (technikai indikátorok, on-chain, makró, esemény feature-ök).
* **LSTM modell**, ami a következő 1 órás **log-return-t** tanulja, ebből számolunk következő árat.
* Rule-based advisor (BUY / HOLD / SELL).
* Egyszerű Flask + Chart.js dashboard.

---

## 📁 Könyvtárstruktúra (lényeges részek)

A projekt gyökere: `crypto_ai_project/`

Fontos mappák és fájlok:

```text
crypto_ai_project/
├── app/
│   ├── dashboard.py          # Flask app (API + HTML dashboard)
│   └── templates/
│       └── dashboard.html    # Frontend UI (Tailwind + Chart.js)
├── data/
│   ├── raw/
│   │   └── bitcoin_kaggle.csv           # Kaggle BTC history (kézzel letöltve)
│   ├── processed/
│   │   ├── market_data.csv              # Binance 1h OHLCV (incrementális)
│   │   ├── market_data_full.csv         # (bootstrap-ból) Kaggle + Binance 1h merge
│   │   ├── onchain_data.csv             # Blockchain.com teljes history napi on-chain
│   │   ├── macro_data.csv               # S&P500 + DXY napi zárók (teljes history)
│   │   ├── sentiment_data.csv           # Rövid (kb. 60 nap) napi sentiment idősor
│   │   ├── news_data.csv                # Max 30 nap nyers hírek (CoinDesk, Reddit, CT)
│   │   ├── training_features_1h.csv     # LSTM train feature store (1h)
│   │   └── training_sentiment_features.csv # Hosszú távú napi sentiment feature store
│   └── runtime/
│       └── market_intraday_1m.csv       # Aznapi 1m Binance OHLCV (naponta felülírva)
├── models/
│   ├── forecast_model.keras             # Keras LSTM modell (log-return target)
│   └── forecast_scalers.pkl             # MinMaxScaler-ek X-re és y-ra (joblib)
├── modules/
│   ├── __init__.py
│   ├── config.py                        # Útvonalak, konstansok, API URL-ek
│   ├── data_collector.py                # Minden, ami adatletöltés
│   ├── feature_engineering.py           # Technikai indikátorok (MA, RSI, stb.)
│   ├── feature_assembler.py             # Market + on-chain + macro + sentiment összejoin
│   ├── sentiment_analyzer.py            # Hírek + Fear&Greed → napi sentiment
│   ├── forecast_model.py                # LSTM train/predict logika (log-return)
│   └── advisor.py                       # Rule-based BUY/HOLD/SELL jelzés
├── bootstrap_market_data.py             # Kaggle + Binance 1H history összefűzés
├── build_training_features.py           # Végső training_features_1h.csv előállítása
├── main.py                              # CLI: update_data, build_features, train, advise
└── venv/                                # Virtuális env (lokális)
```

(A pontos fájlnevek/mappák minimálisan eltérhetnek, de logikailag így néz ki.)

---

## ⚙️ `modules/config.py`

Itt vannak a központi beállítások:

* Alappathok:

  * `BASE_DIR`
  * `DATA_DIR`
  * `MODELS_DIR`
* Konkrét fájlok:

  * `MARKET_DATA_CSV` → `data/processed/market_data.csv`
  * `ONCHAIN_DATA_CSV` → `data/processed/onchain_data.csv`
  * `MACRO_DATA_CSV` → `data/processed/macro_data.csv`
  * `SENTIMENT_DATA_CSV` → `data/processed/sentiment_data.csv`
  * `NEWS_DATA_CSV` → `data/processed/news_data.csv`
  * `TRAINING_FEATURES_CSV` → `data/processed/training_features_1h.csv`
  * `TRAINING_SENTIMENT_FEATURES_CSV` → `data/processed/training_sentiment_features.csv`
  * `MARKET_INTRADAY_1M_CSV` → `data/runtime/market_intraday_1m.csv`
  * `FORECAST_MODEL_PATH` → `models/forecast_model.keras`
  * `FORECAST_SCALER_PATH` → `models/forecast_scalers.pkl`
* API / URL konstansok:

  * `BINANCE_BASE_URL`, `SYMBOL="BTCUSDT"`, `INTERVAL="1h"`
  * `FEAR_GREED_API_URL` (Alternative.me)
  * `BLOCKCHAIN_CHARTS_BASE` (Blockchain.com charts)
  * `COINDESK_RSS_URL`
  * `REDDIT_CRYPTO_RSS_URL`
  * `COINTELEGRAPH_TAG_URLS` (markets, bitcoin)
* Modell paraméterek:

  * `LOOKBACK` (pl. 60 → 60 óra visszatekintő ablak LSTM-hez)

---

## 🧲 Adatgyűjtés – `modules/data_collector.py`

### Binance OHLCV (1H, incrementális)

`update_market_data_csv(symbol=SYMBOL, interval=INTERVAL)`

* Beolvassa a `MARKET_DATA_CSV`-t, ha létezik.
* Megkeresi az utolsó `timestamp`-et.
* Azt követő időponttól hívja a Binance `/api/v3/klines` endpointot.
* Az új gyertyákat hozzáfűzi, duplikátokat timestamp alapján kigyomlálja.
* Mentés: `market_data.csv`, index: `timestamp`, oszlopok: `open, high, low, close, volume`.

### Binance intraday 1m (mai napra)

`update_intraday_minute_data(symbol=SYMBOL)`

* A mai nap 00:00:00 UTC-től kezdve lehúzza az 1 perces BTCUSDT gyertyákat (több batch).
* Cseréli a `MARKET_INTRADAY_1M_CSV` file-t (full rewrite).
* Csak a **mai** napra vonatkozik.

### On-chain (Blockchain.com charts API, teljes history)

`update_onchain_data()`

* Hívott chartok:

  * `n-transactions` → `tx_count`
  * `n-unique-addresses` → `active_addresses`
  * `hash-rate` → `hash_rate`
  * `avg-block-size` → `avg_block_size`
  * `miners-revenue` → `miners_revenue`
* Mindegyiket `timespan=all`-lal húzza.
* Minden chartból DataFrame: index = `timestamp` (UTC napi), `value` → átnevezve.
* Outer join-nal összefésüli.
* Mentés: `onchain_data.csv`

### Makró (Yahoo Finance, teljes history)

`update_macro_data()`

* Ticker mapping:

  * `sp500_close` → `^GSPC`
  * `dxy_close` → `DX-Y.NYB`
* `yf.download(..., period="max", interval="1d", auto_adjust=False)`
* Az Adj Close / Close oszlopból egy Series-t csinál, átnevezi, joinolja.
* Indexet UTC-re lokalizálja.
* Mentés: `macro_data.csv`

---

## 📰 Hírek & Sentiment – `modules/sentiment_analyzer.py`

### Hírforrások

* `fetch_coindesk_rss()` → CoinDesk RSS (limit=100).
* `fetch_reddit_crypto_rss()` → r/CryptoCurrency RSS (limit=100).
* `fetch_cointelegraph_all_tags()` → Cointelegraph:

  * HTML parsolás a tag oldalakból (markets, bitcoin) BeautifulSoup-pal.
  * URL + cím + időpecsét (relatív idő szövegek → `_parse_cointelegraph_relative_date`).
  * Duplikát URL-ek kiszűrése.
  * Csak kb. 30 napon belüli hírek.

### Hírtár frissítés (max 30 nap)

`update_news_store()`

* Beolvassa a létező `NEWS_DATA_CSV`-t (ha hiányzik/hibás, újraépíti).
* Figyel mindenre:

  * `EmptyDataError`
  * rossz formátum
  * timestamp oszlop hiánya
* Lekéri az új híreket (CoinDesk, Reddit, Cointelegraph).
* Összefűzi `df_old` + `df_new`, URL alapján deduplikál.
* Csak az utolsó 30 napot tartja meg:

  * `df_all = df_all[df_all["timestamp"] >= _one_month_ago()]`
* Rendezés timestamp szerint.
* Mentés: `news_data.csv`
  Oszlopok:

  * `timestamp` (UTC, tz-aware)
  * `source`
  * `title`
  * `summary`
  * `url`

**Megjegyzés:** a valóságban az RSS feedek csak 1–2 napnyi cikket adnak vissza, ezért 30 napra vágás ellenére tipikusan **csak az utolsó néhány nap hírei vannak**.

### Cikk-szintű sentiment

`analyze_news_sentiment(df_news)`

* VADER (`SentimentIntensityAnalyzer`) compound score minden cikkre.
* `title` + `summary` → text (NaN → `""`, mindenképp stringgé alakítva).
* Új oszlop: `sentiment` ([-1, 1] tartomány).

### Napi sentiment idősor + long-term training store

`build_sentiment_timeseries()`

* 1. `df_news = update_news_store()`
* 2. `df_scored = analyze_news_sentiment(df_news)`
* 3. Napi aggregáció:

  * `date = timestamp.floor("D")`
  * `groupby("date")`:

    * `news_sentiment = átlag(sentiment)`
    * `news_sentiment_std = szórás(sentiment)`
    * `bullish_ratio = (sentiment > 0).arány`
    * `bearish_ratio = (sentiment < 0).arány`
* 4. Fear & Greed idősor:

  * külön API-hívás (Alternative.me) → napi `fear_greed` értékek.
  * join a napi dataframe-re.
* 5. Rövid runtime sentiment idősor:

  * max ~60 nap → `SENTIMENT_DATA_CSV`
* 6. Hosszú távú training sentiment store:

  * `TRAINING_SENTIMENT_FEATURES_CSV`:

    * beolvassa a régit, index=timestamp
    * hozzáfűzi az új napokat
    * index alapján deduplikál
    * menti vissza

---

## 📐 Feature engineering & training store

### Technikai indikátorok – `modules/feature_engineering.py`

Van egy `add_all_features(df_mkt)` jellegű függvény, ami:

* Bemenet: 1H OHLCV (index: timestamp, oszlopok: open, high, low, close, volume).
* Hozzáad:

  * Alap price feature-ök:

    * `hl_range` (high-low)
    * `oc_diff` (open-close)
    * `ret` (pct_change)
  * Trend indikátorok:

    * MA_7, MA_21, MA_50
    * EMA_12, EMA_26
    * esetleg Hull Moving Average (ha implementálva)
  * Momentum:

    * RSI (14)
    * ROC
  * Volatilitás:

    * rolling STD (7, 30)
    * ATR (Average True Range)
  * Volume-based:

    * OBV
    * Volume change %
* Visszatér: bővített `df_mkt_with_features`.

### Esemény-feature-ök (halving, nagy regulációs/makró események)

A `build_training_features.py` (és/vagy `feature_assembler.py`) során:

* Kézzel összeírt listából generálunk eseményjelzőket, pl.:

  * `event_halving_*` (halving napjai + környezetük, pl. ±30 napban 1)
  * `event_regulation_*` (pl. nagy SEC döntések, Kína-ban, stb.)
  * `event_macro_shock_*` (pl. Covid crash)
* Ezekből 1H-ra vagy 1D-re resample-ölt bináris/időtartam feature-ök keletkeznek, amelyeket hozzájoinolunk a training store-hoz.

### Végső training feature store – `build_training_features.py`

Ez a script:

1. Betölti a **teljes** market history-t (pl. `market_data_full.csv` vagy resample-öl `market_data.csv`-t 1H-ra).
2. Rárakja a technikai indikátorokat (`add_all_features`).
3. Hozzájoinolja:

   * on-chain (`onchain_data.csv`, napi → 1H align, fwd/bwd fill),
   * makró (`macro_data.csv`, napi → 1H align),
   * hosszú távú sentiment (`training_sentiment_features.csv` → napi → 1H align),
   * esemény-feature-öket.
4. Vág:

   * keres közös időtartományt, ahol minden fontos oszlopban van értelmes adat,
   * a nagyon régi időszakokra a sentiment tipikusan semleges (mert nincs history).
5. Tisztít:

   * végtelenek → NaN
   * NaN-ok ésszerű fill/drop logikával.
6. Mentés:

   * `TRAINING_FEATURES_CSV` → `data/processed/training_features_1h.csv`.

---

## 🤖 LSTM modell – `modules/forecast_model.py`

A modell **log-return-t tanul**, nem direkt árat.

### Training adat betöltése

`load_training_data()`

* Betölti a `TRAINING_FEATURES_CSV`-t, index=timestamp.

* Ellenőrzi, hogy van-e `close`.

* Kiszámítja:

  ```python
  df["log_return"] = np.log(df["close"] / df["close"].shift(1))
  df = df.dropna(subset=["log_return"])
  ```

* **Target**: `y = log_return` (N x 1)

* **Features**: `X = df.drop(columns=["log_return"]).values`
  (általában `close` bent marad feature-ként, de igény szerint kivehető).

### Szekvenciák

`build_sequences(X, y, lookback=LOOKBACK)`

* Standard sliding window:

  * X_seq shape: `(N-lookback, lookback, num_features)`
  * y_seq shape: `(N-lookback, 1)`  → a `lookback` utáni log-return.

### Modell tréning

`train_model(epochs=50, batch_size=32, patience=10)`

* MinMaxScaler X-re és y-ra (`FORECAST_SCALER_PATH`-ba mentve).

* Train/test split 90/10.

* Architektúra:

  ```python
  model = Sequential([
      LSTM(128, return_sequences=True, input_shape=(LOOKBACK, num_features)),
      Dropout(0.2),
      LSTM(64),
      Dropout(0.2),
      Dense(1),
  ])
  model.compile(optimizer="adam", loss="mse")
  early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
  model.fit(...)
  ```

* Mentés:

  * modell: `FORECAST_MODEL_PATH` → `models/forecast_model.keras`
  * scalerek: `FORECAST_SCALER_PATH` → `models/forecast_scalers.pkl`

### Modell betöltése

`load_trained_model()`

* `model = load_model(FORECAST_MODEL_PATH, compile=False)`  ← **fontos**
* Scalerek: `joblib.load(FORECAST_SCALER_PATH)`

### Következő ár becslése

`predict_next_close()`

* Újra betölti a `TRAINING_FEATURES_CSV`-et + log-return targetet.

* Az utolsó `LOOKBACK` sorból input ablakot csinál.

* Modellt hívja: predikció **skálázott log-return-re**, majd visszaskálázza.

* Utolsó valós `close`:

  ```python
  last_close = float(df["close"].iloc[-1])
  pred_log_return = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
  predicted_close = last_close * np.exp(pred_log_return)
  ```

* Visszatérés:

  * `predicted_close` (következő órás BTC ár),
  * `last_close`,
  * `last_row` (utolsó sor a df-ből, minden feature-rel).

---

## 💡 Advisor – `modules/advisor.py`

`generate_advice()`

* Meghívja a `predict_next_close()`-t.

* Számolja a relatív változást:

  ```python
  rel_change = (predicted_close - last_close) / last_close
  ```

* Beolvassa a legfrissebb:

  * `fear_greed` értéket,
  * `news_sentiment` értéket (sentiment short idősor legutolsó sora).

* Egy egyszerű szabályrendszer szerint jelzést ad:

  * Ha `rel_change` >> 0 és sentiment/FG is “pozitív” → `BUY`
  * Ha `rel_change` ~0 → `HOLD`
  * Ha `rel_change` << 0 → `SELL`

* Visszaad egy dict-et, pl.:

  ```python
  {
      "signal": "BUY" | "HOLD" | "SELL" | "ERROR",
      "last_close": float,
      "next_price_pred": float,
      "rel_change_pred": float,
      "fear_greed": int | None,
      "news_sentiment": float | None,
      # ha hiba volt:
      "error": "..." (opcionális)
  }
  ```

---

## 🌐 Flask dashboard – `app/dashboard.py` + `templates/dashboard.html`

### Backend (Flask)

`app/dashboard.py`

* `index()` → `/`

  * Rendereli a `dashboard.html`-t.
* `api_state()` → `/api/state`

  * Visszaad JSON-t:

    * `candles_1h` → list of dict:

      * `time`, `open`, `high`, `low`, `close`, `volume`
      * forrás: `MARKET_DATA_CSV` utolsó ~200 sor
    * `intraday_1m` → list of dict:

      * `time`, `price`, `volume`
      * forrás: `MARKET_INTRADAY_1M_CSV` utolsó ~300 sor
    * `sentiment` → dict:

      * `timestamps` (lista ISO dátum)
      * `news_sentiment` (lista float vagy null)
      * `fear_greed` (lista int vagy null)
      * `latest`:

        * `news_sentiment`, `fear_greed`
      * forrás: `SENTIMENT_DATA_CSV`
    * `advice` → a `generate_advice()` kimenete (lásd fent).
  * Ha a modell/scaler hiányzik, `advice.signal = "ERROR"` + `error` mezővel.

Van `create_app()` is, ha WSGI serverrel (gunicorn/uwsgi) akarjuk futtatni.

Indítás lokálisan:

```bash
(venv) python -m app.dashboard
# vagy
(venv) python app/dashboard.py
```

### Frontend (dashboard.html)

* TailwindCSS CDN + Chart.js CDN.
* Három fő panel:

  1. **Jelzés panel**:

     * BUY / HOLD / SELL (vagy ERROR)
     * utolsó záróár
     * következő ár predikció
     * várható % változás
  2. **Hangulat panel**:

     * aktuális Fear & Greed
     * aktuális news_sentiment
     * alatta Chart.js vonaldiagram:

       * news_sentiment (y1)
       * Fear & Greed (y2)
  3. **Intraday (1m)**:

     * egyszerű vonaldiagram az aznapi close-okról.
* Alul: 1H close chart (candlestick helyett most sima close-vonal).
* JS:

  * `/api/state` fetch 60 másodpercenként.
  * `upsertCharts(state)` frissíti/hozza létre a Chart.js grafikonokat.
  * `updateInfoPanels(state)` frissíti a számokat / jelzést.

---

## 🧪 Tipikus futási sorrend

1. **Adatfrissítés** (piaci, on-chain, makró, sentiment):

   ```bash
   python main.py update_data
   ```

2. **Training feature store építés** (ha változtak az adatok / feature-ök):

   ```bash
   python build_training_features.py
   # output: data/processed/training_features_1h.csv
   ```

3. **Modell tanítása**:

   ```bash
   python main.py train --epochs 20
   # vagy simán: python main.py train
   ```

4. **Advisor futtatása CLI-ből**:

   ```bash
   python main.py advise
   ```

5. **Flask dashboard** indítása:

   ```bash
   python -m app.dashboard
   # majd böngészőben: http://localhost:5000/
   ```

---
