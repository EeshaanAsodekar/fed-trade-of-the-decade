### get the fed funds rate
import requests
import pandas as pd

# FRED API key
FRED_API_KEY = "efe3da4f00a3fac72acd1e0dbe68901d"

# Endpoint for Federal Funds Effective Rate (FEDFUNDS)
url = "https://api.stlouisfed.org/fred/series/observations"

# Request data from FRED API
params = {
    "series_id": "FEDFUNDS",
    "observation_start": "2012-01-01",
    "api_key": FRED_API_KEY,
    "file_type": "json"
}
response = requests.get(url, params=params)
data = response.json()

# Extract relevant data
observations = data.get('observations', [])
fed_funds_rate = [{"date": obs["date"], "rate": float(obs["value"])} for obs in observations if obs["value"] != "."]

# Convert to DataFrame
fed_funds_df = pd.DataFrame(fed_funds_rate)

# Save to CSV or process further
fed_funds_df.to_csv("data/raw/fed_funds_rate.csv", index=False)

print(fed_funds_df.head())




### get the fed funds target rate
import requests
import pandas as pd

# FRED API key
FRED_API_KEY = "efe3da4f00a3fac72acd1e0dbe68901d"

# Endpoint for Federal Funds Target Rate (upper bound)
url = "https://api.stlouisfed.org/fred/series/observations"

# Request data from FRED API
params = {
    "series_id": "DFEDTARU",  # Federal Funds Target Rate (Upper Bound)
    "observation_start": "2012-01-01",
    "api_key": FRED_API_KEY,
    "file_type": "json"
}
response = requests.get(url, params=params)
data = response.json()

# Extract relevant data
observations = data.get('observations', [])
fed_funds_target_rate = [{"date": obs["date"], "rate": float(obs["value"])} for obs in observations if obs["value"] != "."]

# Convert to DataFrame
fed_funds_df = pd.DataFrame(fed_funds_target_rate)

# Save to CSV or process further
fed_funds_df.to_csv("data/raw/fed_funds_target_rate.csv", index=False)

print(fed_funds_df.head())




### get fomc meeting dates
# NOTE: for any meeting date e.g. 2024-11-07 the rate change is effective from next
# day viz. 2024-11-08
import os
import pandas as pd

# Directory containing FOMC statements
statements_dir = "data/raw/FOMC/statements"

# List all .txt files in the directory
statement_files = [f for f in os.listdir(statements_dir) if f.endswith(".txt")]

# Extract meeting dates from filenames
fomc_data = []
for filename in statement_files:
    # Extract date from the filename (assumes format YYYY-MM-DD_Statements.txt)
    meeting_date = filename.split("_")[0]
    fomc_data.append({"fomc_meeting_date": meeting_date})

# Create a DataFrame from the extracted dates
fomc_dates_df = pd.DataFrame(fomc_data)
fomc_dates_df.to_csv("data/raw/fomc_meeting_dates.csv")
# Display the DataFrame
print(fomc_dates_df.head())
print(fomc_dates_df.tail())