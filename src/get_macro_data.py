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
    "Unemployment Rate": "UNRATE",
    "CPI (All Urban Consumers)": "CPIAUCSL",
    "GDP Growth Rate": "GDPC1",
    "10-Year Treasury Yield": "DGS10",
    "Fed Funds Rate": "FEDFUNDS",
    "PCE Inflation": "PCEPI",
    "Industrial Production Index": "INDPRO",
    "Total Nonfarm Payrolls": "PAYEMS",
    "Housing Starts": "HOUST",
    "Building Permits": "PERMIT",
    "Consumer Sentiment Index": "UMCSENT",
    "ISM Manufacturing PMI": "MANEMP",
    "ISM Non-Manufacturing PMI": "NMFSL",
    "Trade Balance": "BOPGTB",
    "M2 Money Supply": "M2",
    "30-Year Treasury Yield": "DGS30",
    "Initial Jobless Claims": "ICSA",
}

# Define market data tickers
market_tickers = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Gold": "GC=F",
    "Oil (WTI)": "CL=F",
    "US Dollar Index": "DX-Y.NYB",
    "2-Year Treasury Yield": "^IRX",
    "10-Year Treasury Yield": "^TNX"
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

# Fetch market data using yfinance
def fetch_market_data(ticker, start, end):
    print(f"Downloading market data for {ticker}...")
    data = yf.download(ticker, start=start, end=end)
    return data["Adj Close"]

market_data = {}
for name, ticker in market_tickers.items():
    market_data[name] = fetch_market_data(ticker, start_date, end_date)

# Compute the 2y-10y spread
if "2-Year Treasury Yield" in market_data and "10-Year Treasury Yield" in market_data:
    print("Calculating 2y-10y Treasury Yield Spread...")
    market_data["2y-10y Spread"] = market_data["10-Year Treasury Yield"] - market_data["2-Year Treasury Yield"]

# Combine the market data into a single DataFrame
market_df = pd.DataFrame(market_data)

# Combine macro and market data
combined_df = pd.concat([macro_df, market_df], axis=1)

# Save the combined data to a CSV file
output_file = "data/raw/all_macro_market_data_2012_present.csv"
combined_df.to_csv(output_file, index_label="Date")
print(f"All macro and market data saved to '{output_file}'.")

# Display the first few rows of the combined data
print(combined_df.head())
