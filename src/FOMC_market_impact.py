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
    
    # S&P 500 Indexes
    "Financials (SP500)": "^SP500-40",
    "Real Estate (SP500)": "^SP500-60",
    "Utilities (SP500)": "^SP500-55",
    "Consumer Discretionary (SP500)": "^SP500-25",
    "Consumer Staples (SP500)": "^SP500-30",
    "Technology (SP500)": "^SP500-45",
    "Industrials (SP500)": "^SP500-20",

    # ETFs
    "Financials (ETF)": "XLF",
    "Real Estate (ETF)": "XLRE",
    "Utilities (ETF)": "XLU",
    "Consumer Discretionary (ETF)": "XLY",
    "Consumer Staples (ETF)": "XLP",
    "Technology (ETF)": "XLK",
    "Industrials (ETF)": "XLI",
    "Energy (ETF)": "XLE",

    "VIX": "^VIX",
    
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Oil (WTI)": "CL=F",
    
    "US Dollar Index": "DX-Y.NYB",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "USDJPY=X",
}

# Specify the time range
start_date = "2016-01-01"
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
print("nulls", df_interpolated.isna().sum())

df_interpolated.dropna(axis=0,inplace=True)

print(df_interpolated.head())
print(df_interpolated.shape)
print("\nFINAL NULLS: ",df_interpolated.isna().sum().sum())
print(df_interpolated.columns)

df_interpolated.to_csv("trade_effect_market_data.csv")



### getting changes for each FOMC meeting rate decision
# Load the rate_moves and df_interpolated data
rate_moves = pd.read_csv("data/processed/rate_moves.csv", parse_dates=["date"])

rate_moves = rate_moves[rate_moves["date"] >= "2016-01-01"]

# Initialize a dictionary to store computed percentage changes
percentage_changes = {col: [] for col in df_interpolated.columns}

# Loop through each FOMC meeting date
for meeting_date in rate_moves["date"]:
    # Define the analysis window (T-1 to T+3)
    start_date = meeting_date - pd.Timedelta(days=1)
    end_date = meeting_date + pd.Timedelta(days=3)
    
    print("*************************")
    print(meeting_date)
    print(start_date)
    print(end_date)
    print("*************************")

    # Extract the subset of df_interpolated for the window
    window_data = df_interpolated.loc[start_date:end_date]
    
    # Compute the percentage change from T-1 to T+3
    if len(window_data) >= 2:
        changes = (window_data.iloc[-1] - window_data.iloc[0]) / window_data.iloc[0]
    else:
        changes = pd.Series([None] * len(df_interpolated.columns), index=df_interpolated.columns)
    
    # Append the changes to the corresponding column
    for col in df_interpolated.columns:
        percentage_changes[col].append(changes[col])

# Add the computed percentage changes as new columns to rate_moves
for col, changes in percentage_changes.items():
    rate_moves[col + "_pct_change"] = changes

print(">>> nulls: ",rate_moves.isna().sum().sum())

# Save the updated rate_moves dataframe
rate_moves.to_csv("rate_moves_updated.csv", index=False)

# Display the first few rows of the updated rate_moves DataFrame
print(rate_moves.head())
print(rate_moves.columns)