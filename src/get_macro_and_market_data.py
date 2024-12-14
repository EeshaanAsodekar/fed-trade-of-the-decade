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
    "Core CPI (Ex Food & Energy)": "CPILFESL",
    "PCE Inflation": "PCEPI",
    "Core PCE Inflation (Ex Food & Energy)": "PCEPILFE",
    "Trimmed Mean PCE": "PCETRIM12M159SFRBDAL",
    "Median CPI": "MEDCPIM158SFRBCLE",
    "Real GDP": "A191RL1Q225SBEA",
    "GDP Growth Rate": "GDPC1",
    "10-Year Treasury Yield": "DGS10",
    "2-Year Treasury Yield": "DGS2",
    "Industrial Production Index": "INDPRO",
    "Total Nonfarm Payrolls": "PAYEMS",
    "Housing Starts": "HOUST",
    "Building Permits": "PERMIT",
    "Consumer Sentiment Index": "UMCSENT",
    "ISM Manufacturing PMI": "MANEMP",
    "ISM Non-Manufacturing PMI": "NMFSL",
    "Trade Balance": "BOPGTB",
    "M2 Money Supply": "M2SL",
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
}

# Specify the time range
start_date = "2011-01-01"
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

# Save the combined data to a CSV file
output_file = "data/raw/all_macro_market_data_2012_present.csv"
combined_df.to_csv(output_file, index_label="Date")
print(f"All macro and market data saved to '{output_file}'.")

# Display the first few rows of the combined data
print(combined_df.head())

### linearly interpollating the macro and market data
# Load the data
df = pd.read_csv(r"data\raw\all_macro_market_data_2012_present.csv")

# Ensure the 'Date' column is a datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

df_interpolated = df.interpolate(method='linear', axis=0)

print(df_interpolated.head())
print(df_interpolated.shape)
print(df_interpolated.isna().sum().sum())

df_interpolated.dropna(axis=0,inplace=True)

print(df_interpolated.head())
print(df_interpolated.shape)
print(df_interpolated.isna().sum().sum())

# Compute annualized percentage changes for the specified macro variables
annual_change_vars = [
    "CPI (All Urban Consumers)",
    "Core CPI (Ex Food & Energy)",
    "PCE Inflation",
    "Core PCE Inflation (Ex Food & Energy)",
]
for var in annual_change_vars:
    if var in df_interpolated.columns:
        # Compute YoY percentage change
        df_interpolated[f"{var} YoY Change"] = (
            (df_interpolated[var] / df_interpolated[var].shift(252) - 1) * 100
        )

# Drop rows before 2012-01-01- don't want 2011 data in final dataset
start_date = "2012-01-01"
df_interpolated = df_interpolated[df_interpolated.index >= start_date]

# Save the final truncated dataset
output_file = "data/processed/interpolated_all_macro_market_data_2012_present.csv"
df_interpolated.to_csv(output_file, index_label="Date")




### visualizing for sanity check
import matplotlib.pyplot as plt

# List of columns to plot
columns_to_plot = [
    "CPI (All Urban Consumers)",
    "Core CPI (Ex Food & Energy)",
    "PCE Inflation",
    "Core PCE Inflation (Ex Food & Energy)"
]

for column in columns_to_plot:
    yoy_column = f"{column} YoY Change"
    
    if column in df.columns and yoy_column in df_interpolated.columns:
        plt.figure(figsize=(12, 6))
        
        # Plot original data points
        plt.scatter(
            df.index, 
            df[column], 
            color='red', 
            label='Original Data Points', 
            s=50,  # Size of red dots
            alpha=0.7
        )
        
        # Plot interpolated data as a line
        plt.plot(
            df_interpolated.index, 
            df_interpolated[column], 
            color='blue', 
            label='Interpolated Data', 
            linewidth=1.5
        )
        
        # Plot YoY change on a secondary y-axis
        ax = plt.gca()  # Get current axis
        ax2 = ax.twinx()  # Create a secondary y-axis
        
        ax2.plot(
            df_interpolated.index, 
            df_interpolated[yoy_column], 
            color='green', 
            linestyle='--', 
            label=f'{column} YoY Change', 
            linewidth=1.5, 
            alpha=0.7
        )
        ax2.set_ylabel('YoY Change (%)', fontsize=12, color='green')
        
        # Adjust appearance
        ax2.tick_params(axis='y', colors='green')
        
        # Add labels and title
        ax.set_title(f'Original, Interpolated, and YoY Change for {column}', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(column, fontsize=12, color='blue')
        ax.tick_params(axis='y', colors='blue')
        ax.legend(loc='upper left', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)
        
        # Grid and x-axis formatting
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        # Show plot
        plt.tight_layout()
        plt.show()
    else:
        print(f"Column '{column}' or its YoY Change column not found in the dataset.")
