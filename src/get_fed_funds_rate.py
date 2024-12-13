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