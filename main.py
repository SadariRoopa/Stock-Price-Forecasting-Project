import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def fetch_stock_data(symbol="AAPL", period="1y"):
    # Download stock data without group_by to avoid multi-index columns
    df = yf.download(symbol, period=period)

    # If columns have MultiIndex (like ('Close', 'AAPL')), flatten them:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]

    # Verify columns after flattening
    # We expect columns like Open, High, Low, Close etc. 
    # If appended by symbol, fix selectors later in app.py accordingly.

    # Select relevant columns - handle potential symbol suffixes (e.g. Close_AAPL)
    close_col = next((col for col in df.columns if col.startswith('Close')), None)
    open_col = next((col for col in df.columns if col.startswith('Open')), None)
    high_col = next((col for col in df.columns if col.startswith('High')), None)
    low_col = next((col for col in df.columns if col.startswith('Low')), None)
    volume_col = next((col for col in df.columns if col.startswith('Volume')), None)

    if not all([close_col, open_col, high_col, low_col, volume_col]):
        raise ValueError(f"Missing expected price/volume columns. Found columns: {df.columns.tolist()}")

    df = df[[open_col, high_col, low_col, close_col, volume_col]]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # rename standardized

    # Add technical indicators
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()

    df.dropna(inplace=True)
    df.reset_index(inplace=True)  # Ensure 'Date' as a column

    return df


def prepare_data(df):
    features = ['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'Volatility']
    X = df[features]
    y = df['Close']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def train_models(df):
    X, y = prepare_data(df)
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        results[name] = {
            'model': model,
            'mse': mse,
            'predictions': predictions,
            'actual': y_test
        }

    return results

