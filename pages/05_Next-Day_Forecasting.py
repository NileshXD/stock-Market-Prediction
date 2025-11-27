import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from millify import millify

st.subheader('Next-Day Forecasting (RandomForest & Linear Regression)')

csv = pd.read_csv('symbols.csv')
symbol = csv['Symbol'].tolist()
for i in range(len(symbol)):
    symbol[i] = symbol[i] + ".NS"

# Sidebar / selectbox
ticker = st.selectbox(
    'Enter or Choose NSE listed Stock Symbol',
    symbol,
    index=symbol.index('TRIDENT.NS')
)

def check_data_length(df, min_rows=200):
    """Ensure dataset has enough rows to train ML models."""
    if df is None or df.empty:
        st.error("❌ No data returned for this ticker. Try another symbol.")
        st.stop()

    if len(df) < min_rows:
        st.error(f"❌ Not enough data to train the forecasting model.\n"
                 f"Required: {min_rows} rows\n"
                 f"Found: {len(df)} rows\n\n"
                 f"Try another ticker with longer history.")
        st.stop()


def load_price_data(ticker: str) -> pd.DataFrame:
    """Download last 5 years of data safely and handle empty/missing columns."""
    start = dt.datetime.today() - dt.timedelta(5 * 365)
    end = dt.datetime.today()

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column"
    )

    # ─────────────────────────────────────────────────────────
    # 1️⃣ If dataframe empty → Stop immediately
    # ─────────────────────────────────────────────────────────
    if df is None or df.empty:
        raise KeyError(f"No price data returned for ticker: {ticker}")

    # ─────────────────────────────────────────────────────────
    # 2️⃣ Flatten MultiIndex columns if present
    # ─────────────────────────────────────────────────────────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # ─────────────────────────────────────────────────────────
    # 3️⃣ Fix missing Close / Adj Close
    # ─────────────────────────────────────────────────────────

    # Create Close from Adj Close
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            # create synthetic Close
            df["Close"] = np.nan

    # Create Adj Close from Close
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    # ─────────────────────────────────────────────────────────
    # 4️⃣ Final fallback for missing OHLC
    # ─────────────────────────────────────────────────────────
    for col in ["Open", "High", "Low", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    return df



def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple tabular features for next-day close prediction."""
    data = df.copy()
    data = data.sort_values("Date")

    # Basic features
    data["Return_1d"] = data["Close"].pct_change()
    data["MA_5"] = data["Close"].rolling(window=5).mean()
    data["MA_10"] = data["Close"].rolling(window=10).mean()
    data["MA_20"] = data["Close"].rolling(window=20).mean()
    data["Vol_5"] = data["Close"].rolling(window=5).std()
    data["Vol_20"] = data["Close"].rolling(window=20).std()

    # Target: next day's Close
    data["Target_Close_next"] = data["Close"].shift(-1)

    data = data.dropna().reset_index(drop=True)
    return data

def train_models(data: pd.DataFrame):
    """Train RandomForest & LinearRegression on historical data."""
    feature_cols = ["Close", "Return_1d", "MA_5", "MA_10", "MA_20", "Vol_5", "Vol_20"]

    X = data[feature_cols].values
    y = data["Target_Close_next"].values

    # use last ~60 days as test set
    test_size = min(60, len(data) // 5)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # predictions
    rf_pred = rf.predict(X_test)
    lr_pred = lr.predict(X_test)

    metrics = {
        "rf": {
            "rmse": mean_squared_error(y_test, rf_pred, squared=False),
            "mae": mean_absolute_error(y_test, rf_pred),
            "r2": r2_score(y_test, rf_pred),
            "y_pred": rf_pred,
        },
        "lr": {
            "rmse": mean_squared_error(y_test, lr_pred, squared=False),
            "mae": mean_absolute_error(y_test, lr_pred),
            "r2": r2_score(y_test, lr_pred),
            "y_pred": lr_pred,
        },
        "y_test": y_test,
        "dates_test": data["Date"].iloc[-test_size:].values,
        "feature_cols": feature_cols,
        "rf_model": rf,
        "lr_model": lr,
    }
    return metrics

def forecast_next_day(models_info, latest_row: pd.Series):
    """Use trained models to predict the next day's Close."""
    feature_cols = models_info["feature_cols"]
    X_next = latest_row[feature_cols].values.reshape(1, -1)

    rf_pred = models_info["rf_model"].predict(X_next)[0]
    lr_pred = models_info["lr_model"].predict(X_next)[0]
    return rf_pred, lr_pred

# === Main run ===
df = load_price_data(ticker)

# 1️⃣ Check raw data
check_data_length(df, min_rows=200)

feat_df = build_features(df)

# 2️⃣ Check engineered data
check_data_length(feat_df, min_rows=150)

st.write(f"Using data from **{df['Date'].min()}** to **{df['Date'].max()}**.")



if len(feat_df) < 100:
    st.error("Not enough data to train the forecasting models.")
else:
    models_info = train_models(feat_df)

    # Metrics
    st.markdown("### Model Performance (last ~60 days)")
    col1, col2, col3 = st.columns(3)
    col1.metric("RandomForest RMSE", f"{models_info['rf']['rmse']:.2f}")
    col2.metric("RandomForest MAE", f"{models_info['rf']['mae']:.2f}")
    col3.metric("RandomForest R²", f"{models_info['rf']['r2']:.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("LinearReg RMSE", f"{models_info['lr']['rmse']:.2f}")
    col5.metric("LinearReg MAE", f"{models_info['lr']['mae']:.2f}")
    col6.metric("LinearReg R²", f"{models_info['lr']['r2']:.3f}")

    # Plot actual vs predictions
    dates_test = models_info["dates_test"]
    y_test = models_info["y_test"]
    rf_pred = models_info["rf"]["y_pred"]
    lr_pred = models_info["lr"]["y_pred"]

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(x=dates_test, y=y_test, name="Actual Close")
    )
    fig.add_trace(
        go.Scatter(x=dates_test, y=rf_pred, name="RF Predicted")
    )
    fig.add_trace(
        go.Scatter(x=dates_test, y=lr_pred, name="LR Predicted")
    )
    fig.update_layout(
        height=600,
        title_text=f"Actual vs Predicted Next-Day Close: {ticker}",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Next-day forecast
    latest_row = feat_df.iloc[-1]
    rf_next, lr_next = forecast_next_day(models_info, latest_row)
    last_close = feat_df["Close"].iloc[-1]

    st.markdown("### Next-Day Forecast")
    c1, c2, c3 = st.columns(3)
    c1.metric("Last Close", f"₹{millify(last_close, 2)}")
    c2.metric("RF Next-Day Close", f"₹{millify(rf_next, 2)}")
    c3.metric("LR Next-Day Close", f"₹{millify(lr_next, 2)}")
