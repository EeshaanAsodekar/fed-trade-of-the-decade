from fredapi import Fred
import pandas as pd
import datetime
import yfinance as yf

# Replace with your FRED API Key
FRED_API_KEY = "efe3da4f00a3fac72acd1e0dbe68901d"

# Initialize the FRED API
fred = Fred(api_key=FRED_API_KEY)

# Define the key macro variables and their FRED series IDs
macro_variables = {
    "10-Year Treasury Yield": "DGS10",
    "2-Year Treasury Yield": "DGS2",
    "30-Year Treasury Yield": "DGS30",
}

# Define market data tickers
market_tickers = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Gold": "GC=F",
    "Oil (WTI)": "CL=F",
    "US Dollar Index": "DX-Y.NYB",
}

# Specify the time range
start_date = "2012-01-01"
end_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Fetch macroeconomic data from FRED
macro_data = {}
for name, series_id in macro_variables.items():
    try:
        print(f"Downloading macro data for {name}...")
        macro_data[name] = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    except ValueError as e:
        print(f"Error fetching {name} (Series ID: {series_id}): {e}")
# Combine the macro data into a single DataFrame
macro_df = pd.DataFrame(macro_data)

# After fetching the data for all variables
macro_df["2-10 Spread"] = macro_df["10-Year Treasury Yield"] - macro_df["2-Year Treasury Yield"]

# Fetch market data using yfinance
def fetch_market_data(ticker, start, end):
    print(f"Downloading market data for {ticker}...")
    data = yf.download(ticker, start=start, end=end)
    return data["Adj Close"]

market_data = {}
for name, ticker in market_tickers.items():
    market_data[name] = fetch_market_data(ticker, start_date, end_date)

# Combine the market data into a single DataFrame
market_df = pd.DataFrame(market_data)

# Combine macro and market data
combined_df = pd.concat([macro_df, market_df], axis=1)

print(combined_df.head())

### linearly interpollating the macro and market data
df = combined_df

df_interpolated = df.interpolate(method='linear', axis=0)

print(df_interpolated.head())
print(df_interpolated.shape)
print(df_interpolated.isna().sum().sum())

df_interpolated.dropna(axis=0,inplace=True)

print(df_interpolated.head())
print(df_interpolated.shape)
print(df_interpolated.isna().sum().sum())
print(df_interpolated.columns)

df_interpolated.to_csv("trade_effect_market_data.csv")