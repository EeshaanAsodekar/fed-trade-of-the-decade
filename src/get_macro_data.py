from fredapi import Fred
import pandas as pd
import datetime

# Replace with your FRED API Key
FRED_API_KEY = "efe3da4f00a3fac72acd1e0dbe68901d"

# Initialize the FRED API
fred = Fred(api_key=FRED_API_KEY)

# Define the key macro variables and their FRED series IDs
macro_variables = {
    "Unemployment Rate": "UNRATE",                # Unemployment Rate
    "CPI (All Urban Consumers)": "CPIAUCSL",      # Consumer Price Index (CPI)
    "GDP Growth Rate": "GDPC1",                   # Real GDP
    "10-Year Treasury Yield": "DGS10",            # 10-Year Treasury Yield
    "Fed Funds Rate": "FEDFUNDS",                 # Effective Federal Funds Rate
    "PCE Inflation": "PCEPI",                     # Personal Consumption Expenditures Price Index
    "Industrial Production Index": "INDPRO",      # Industrial Production
    "Total Nonfarm Payrolls": "PAYEMS"            # Total Nonfarm Payroll Employment
}

# Specify the time range
start_date = "2021-01-01"
end_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Fetch the data for each variable and store in a dictionary
data = {}
for name, series_id in macro_variables.items():
    print(f"Downloading data for {name}...")
    data[name] = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

# Combine the data into a single DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("data/raw/fed_macro_variables_2021_present.csv", index_label="Date")
print("Macro variables data saved to 'fed_macro_variables_2021_present.csv'.")

# Display the first few rows of the data
print(df.head())
