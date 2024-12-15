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

import matplotlib.pyplot as plt

# validating the 2y 10y and 2y 10y spread - SANITY CHECK
selected_columns = ["10-Year Treasury Yield_pct_change", 
                    "2-Year Treasury Yield_pct_change", 
                    "2y-10y Spread_pct_change"]
plt.figure(figsize=(12, 6))
for col in selected_columns:
    if col in rate_moves:
        plt.scatter(rate_moves['date'], rate_moves[col], label=col, s=10)  # Plot as points

plt.title('Percentage Changes in Selected Market Variables')
plt.xlabel('Date')
plt.ylabel('Percentage Change')
plt.legend(loc='best', fontsize='small')
plt.grid(True)
# Format the x-axis to show dates in mm-dd-yy format
plt.xticks(rate_moves['date'], rate_moves['date'].dt.strftime('%m-%d-%y'), rotation=45, fontsize=8)
plt.tight_layout()
plt.show()




# plotting by quintiles
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# List of columns to visualize
pct_change_columns = [
    '10-Year Treasury Yield_pct_change', 
    '2-Year Treasury Yield_pct_change',
    '30-Year Treasury Yield_pct_change', 
    '2-10 Spread_pct_change',
    'S&P 500_pct_change', 
    'Nasdaq_pct_change', 
    'Financials (SP500)_pct_change',
    'Real Estate (SP500)_pct_change', 
    'Utilities (SP500)_pct_change',
    'Consumer Discretionary (SP500)_pct_change', 
    'Consumer Staples (SP500)_pct_change',
    'Technology (SP500)_pct_change', 
    'Industrials (SP500)_pct_change',
    'Gold_pct_change', 
    'Silver_pct_change', 
    'Copper_pct_change',
    'Oil (WTI)_pct_change', 
    'US Dollar Index_pct_change',
    'EUR/USD_pct_change', 
    'USD/JPY_pct_change',
]

# Create a plot for each column grouped by 'rate_change'
for column in pct_change_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='rate_change', y=column, data=rate_moves, palette="Set2")
    plt.title(f"{column} by Rate Change")
    plt.xlabel("Rate Change")
    plt.ylabel("Percentage Change")
    plt.tight_layout()
    # plt.savefig(f"plt_{column}_change.png")
    plt.close()

print(rate_moves.columns)


print("SHAPE >>> ",rate_moves.shape)
print("NULLS >>> ",rate_moves.isna().sum().sum())

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 1: Create the labels for "Hike", "Hold", and "Cut"
rate_moves["rate_label"] = rate_moves["rate_change"].apply(lambda x: "hike" if x > 0 else ("cut" if x < 0 else "hold"))

# Convert labels to numeric values for logistic regression
label_mapping = {"hike": 0, "hold": 1, "cut": 2}
rate_moves["rate_label_numeric"] = rate_moves["rate_label"].map(label_mapping)

# Step 2: Prepare the feature set (market variables)
# List of expected pct_change columns
expected_pct_columns = [
    '10-Year Treasury Yield_pct_change', '2-Year Treasury Yield_pct_change',
    '30-Year Treasury Yield_pct_change', '2-10 Spread_pct_change',
    'S&P 500_pct_change', 'Nasdaq_pct_change', 'Financials (SP500)_pct_change',
    'Real Estate (SP500)_pct_change', 'Utilities (SP500)_pct_change',
    'Consumer Discretionary (SP500)_pct_change', 'Consumer Staples (SP500)_pct_change',
    'Technology (SP500)_pct_change', 'Industrials (SP500)_pct_change',
    'Financials (ETF)_pct_change', 'Real Estate (ETF)_pct_change',
    'Utilities (ETF)_pct_change', 'Consumer Discretionary (ETF)_pct_change',
    'Consumer Staples (ETF)_pct_change', 'Technology (ETF)_pct_change',
    'Industrials (ETF)_pct_change', 'Energy (ETF)_pct_change', 'VIX_pct_change',
    'Gold_pct_change', 'Silver_pct_change', 'Copper_pct_change',
    'Oil (WTI)_pct_change', 'US Dollar Index_pct_change',
    'EUR/USD_pct_change', 'USD/JPY_pct_change',
]

feature_columns = [col for col in expected_pct_columns]
print("shit agfter",rate_moves.columns)
X = rate_moves[expected_pct_columns]

y = rate_moves["rate_label_numeric"]  # Target variable

# Standardize the features for regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Fit Multinomial Logistic Regression
log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
log_reg.fit(X_scaled, y)

# Step 4: Extract Portfolio Weights
# The coefficients for each label (hike, hold, cut) are stored in log_reg.coef_
portfolio_weights = pd.DataFrame(log_reg.coef_.T, columns=["hike_weight", "hold_weight", "cut_weight"], index=feature_columns)

# Normalize weights to sum to 1 for each label
portfolio_weights["hike_weight_normalized"] = portfolio_weights["hike_weight"] / np.sum(np.abs(portfolio_weights["hike_weight"]))
portfolio_weights["hold_weight_normalized"] = portfolio_weights["hold_weight"] / np.sum(np.abs(portfolio_weights["hold_weight"]))
portfolio_weights["cut_weight_normalized"] = portfolio_weights["cut_weight"] / np.sum(np.abs(portfolio_weights["cut_weight"]))

# Save portfolio weights for reference
portfolio_weights.to_csv("portfolio_weights_multinomial.csv", index_label="Variable")

# Display the portfolio weights
print(portfolio_weights)

import statsmodels.api as sm

# Initialize a dictionary to store coefficients
coefficients = {}

# Loop through each pct_change column and fit a linear regression
for column in pct_change_columns:
    X = rate_moves["rate_change"]  # Independent variable
    y = rate_moves[column]  # Dependent variable
    
    # Add a constant for the intercept
    X = sm.add_constant(X)
    
    # Fit the regression model
    model = sm.OLS(y, X).fit()
    
    # Extract the coefficient for 'rate_change'
    coefficients[column] = model.params["rate_change"]

# Create a DataFrame for visualization
coeff_df = pd.DataFrame.from_dict(coefficients, orient="index", columns=["Coefficient"])
coeff_df = coeff_df.sort_values(by="Coefficient")

# Plot the coefficients
plt.figure(figsize=(12, 8))
coeff_df["Coefficient"].plot(kind="bar", color=coeff_df["Coefficient"].apply(lambda x: "red" if x < 0 else "green"))
plt.title("Coefficients of Market Variables vs. Rate Change", fontsize=14)
plt.xlabel("Market Variable", fontsize=12)
plt.ylabel("Coefficient", fontsize=12)
plt.xticks(rotation=90)
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.show()

print(rate_moves.columns)






### portolfio vizl 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Assuming you already have your DataFrame 'rate_moves'
# Columns include 'date', 'rate_change', and various '%_pct_change' columns.

# Define which columns are features (independent variables: market instrument returns)
feature_cols = [
    '10-Year Treasury Yield_pct_change',
    '2-Year Treasury Yield_pct_change',
    '30-Year Treasury Yield_pct_change',
    '2-10 Spread_pct_change',
    'S&P 500_pct_change',
    'Nasdaq_pct_change',
    'Financials (SP500)_pct_change',
    'Real Estate (SP500)_pct_change',
    'Utilities (SP500)_pct_change',
    'Consumer Discretionary (SP500)_pct_change',
    'Consumer Staples (SP500)_pct_change',
    'Technology (SP500)_pct_change',
    'Industrials (SP500)_pct_change',
    'Gold_pct_change',
    'Silver_pct_change',
    'Copper_pct_change',
    'Oil (WTI)_pct_change',
    'US Dollar Index_pct_change',
    'EUR/USD_pct_change',
    'USD/JPY_pct_change'
]

# Clean data
df = rate_moves.copy()
df = df.dropna(subset=feature_cols + ['rate_change'])
df = df.reset_index(drop=True)

# Create scenario labels:
# Hike scenario: rate_change > 0
# Cut scenario: rate_change < 0
df['is_hike'] = (df['rate_change'] > 0).astype(int)
df['is_cut'] = (df['rate_change'] < 0).astype(int)

X = df[feature_cols].values  # Independent variables: market returns

# -------------------------------------------------------
# Portfolio for Hike scenario
# Dependent variable (Y): is_hike (1 if hike occurred, 0 otherwise)
# We want a portfolio that replicates a digital payoff for hikes.
# -------------------------------------------------------
y_hike = df['is_hike'].values

hike_model = LinearRegression()
hike_model.fit(X, y_hike)

hike_weights = hike_model.coef_
hike_intercept = hike_model.intercept_

print("=== Hike Scenario Portfolio ===")
print("Intercept:", hike_intercept)
hike_portfolio = pd.Series(hike_weights, index=feature_cols)
print("Hike Portfolio Weights:")
print(hike_portfolio)

# Predict the "payoff" for each scenario using the hike weights
df['portfolio_hike_pred'] = hike_model.predict(X)

# Performance statistics for the hike portfolio
hike_returns_on_hike_days = df.loc[df['is_hike'] == 1, 'portfolio_hike_pred']
hike_returns_on_non_hike_days = df.loc[df['is_hike'] == 0, 'portfolio_hike_pred']

median_hike_return_on_hike_days = hike_returns_on_hike_days.median()
min_hike_return_on_hike_days = hike_returns_on_hike_days.min()
max_hike_return_on_hike_days = hike_returns_on_hike_days.max()

median_hike_return_on_non_hike_days = hike_returns_on_non_hike_days.median()
min_hike_return_on_non_hike_days = hike_returns_on_non_hike_days.min()
max_hike_return_on_non_hike_days = hike_returns_on_non_hike_days.max()

print("\nHike Portfolio Performance on Hike Days:")
print("Median:", median_hike_return_on_hike_days)
print("Minimum:", min_hike_return_on_hike_days)
print("Maximum:", max_hike_return_on_hike_days)

print("\nHike Portfolio Performance on Non-Hike Days:")
print("Median:", median_hike_return_on_non_hike_days)
print("Minimum:", min_hike_return_on_non_hike_days)
print("Maximum:", max_hike_return_on_non_hike_days)


# -------------------------------------------------------
# Portfolio for Cut scenario
# Dependent variable (Y): is_cut (1 if cut occurred, 0 otherwise)
# We want a portfolio that replicates a digital payoff for cuts.
# -------------------------------------------------------
y_cut = df['is_cut'].values

cut_model = LinearRegression()
cut_model.fit(X, y_cut)

cut_weights = cut_model.coef_
cut_intercept = cut_model.intercept_

print("\n=== Cut Scenario Portfolio ===")
print("Intercept:", cut_intercept)
cut_portfolio = pd.Series(cut_weights, index=feature_cols)
print("Cut Portfolio Weights:")
print(cut_portfolio)

df['portfolio_cut_pred'] = cut_model.predict(X)

# Performance statistics for the cut portfolio
cut_returns_on_cut_days = df.loc[df['is_cut'] == 1, 'portfolio_cut_pred']
cut_returns_on_non_cut_days = df.loc[df['is_cut'] == 0, 'portfolio_cut_pred']

median_cut_return_on_cut_days = cut_returns_on_cut_days.median()
min_cut_return_on_cut_days = cut_returns_on_cut_days.min()
max_cut_return_on_cut_days = cut_returns_on_cut_days.max()

median_cut_return_on_non_cut_days = cut_returns_on_non_cut_days.median()
min_cut_return_on_non_cut_days = cut_returns_on_non_cut_days.min()
max_cut_return_on_non_cut_days = cut_returns_on_non_cut_days.max()

print("\nCut Portfolio Performance on Cut Days:")
print("Median:", median_cut_return_on_cut_days)
print("Minimum:", min_cut_return_on_cut_days)
print("Maximum:", max_cut_return_on_cut_days)

print("\nCut Portfolio Performance on Non-Cut Days:")
print("Median:", median_cut_return_on_non_cut_days)
print("Minimum:", min_cut_return_on_non_cut_days)
print("Maximum:", max_cut_return_on_non_cut_days)

# -------------------------------------------
# Visualization
# -------------------------------------------

# Distribution of the hike portfolio predicted payoff
plt.figure(figsize=(10,6))
sns.kdeplot(df.loc[df['is_hike'] == 1, 'portfolio_hike_pred'], label='Hike Days', shade=True)
sns.kdeplot(df.loc[df['is_hike'] == 0, 'portfolio_hike_pred'], label='Non-Hike Days', shade=True)
plt.title('Hike Scenario Portfolio Predicted Payoff Distribution')
plt.xlabel('Predicted Portfolio Payoff')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# Distribution of the cut portfolio predicted payoff
plt.figure(figsize=(10,6))
sns.kdeplot(df.loc[df['is_cut'] == 1, 'portfolio_cut_pred'], label='Cut Days', shade=True)
sns.kdeplot(df.loc[df['is_cut'] == 0, 'portfolio_cut_pred'], label='Non-Cut Days', shade=True)
plt.title('Cut Scenario Portfolio Predicted Payoff Distribution')
plt.xlabel('Predicted Portfolio Payoff')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# Actual vs predicted comparison for hikes
plt.figure(figsize=(8,6))
plt.scatter(df['is_hike'], df['portfolio_hike_pred'], alpha=0.7)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('Actual Hike Indicator')
plt.ylabel('Predicted Portfolio Value (Hike Portfolio)')
plt.title('Hike Scenario: Actual vs Predicted')
plt.tight_layout()
plt.show()

# Actual vs predicted comparison for cuts
plt.figure(figsize=(8,6))
plt.scatter(df['is_cut'], df['portfolio_cut_pred'], alpha=0.7)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('Actual Cut Indicator')
plt.ylabel('Predicted Portfolio Value (Cut Portfolio)')
plt.title('Cut Scenario: Actual vs Predicted')
plt.tight_layout()
plt.show()

# This setup ensures:
# - The scenario indicator (is_hike or is_cut) is the dependent variable (Y).
# - The market returns (%_pct_change columns) are the independent variables (X).
# The linear model finds weights that approximate the binary event using a linear combination of returns.
# The resulting weights form a "replicating portfolio."
