# import pandas as pd
# import matplotlib.pyplot as plt

# # -----------------------------
# # 1. Load the DataFrames
# # -----------------------------
# # Load the BBG dataset
# bbg_df = pd.read_excel(r"data\weekly_FOMC_Sentiment_data.xlsx")
# print("BBG dataset:")
# print(bbg_df.head())
# print(bbg_df.tail())

# # Load your analysis dataset
# df = pd.read_csv("final_analysis.csv")
# print("My analysis:")
# print(df.head())
# print(df.tail())

# econ_index_df = pd.read_csv("econ_analysis.csv")
# print("econ analysis df")
# print(econ_index_df.head())
# print(econ_index_df.tail())
# # need to rename  the hawkishness_index column of econ_index_df to econ_hawkishness_index...

# # -----------------------------
# # 2. Process Dates
# # -----------------------------
# # Convert the 'Date' column in bbg_df to datetime and drop invalid dates
# bbg_df['Date'] = pd.to_datetime(bbg_df['Date'], errors='coerce')
# bbg_df = bbg_df.dropna(subset=['Date'])
# # Sort bbg_df by date in ascending order
# bbg_df = bbg_df.sort_values(by='Date').reset_index(drop=True)

# # Convert the 'date' column in df to datetime and sort
# df['date'] = pd.to_datetime(df['date'], errors='coerce')
# df = df.sort_values(by='date').reset_index(drop=True)

# # -----------------------------
# # 3. Restrict Data to the Date Range in df
# # -----------------------------
# # Determine the date range based on the df data (rate_change)
# min_date = df['date'].min()
# max_date = df['date'].max()

# # Filter the BBG dataframe to only include dates within the rate_change date range
# bbg_df_filtered = bbg_df[(bbg_df['Date'] >= min_date) & (bbg_df['Date'] <= max_date)]

# # -----------------------------
# # 4. Plot the Data
# # -----------------------------
# fig, ax1 = plt.subplots(figsize=(12, 6))

# # Plot rate_change on the primary y-axis (left side)
# ax1.plot(df['date'], df['rate_change'], label='Rate Change', color='red',
#          linestyle='-', marker='o')
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Rate Change', color='red')
# ax1.tick_params(axis='y', labelcolor='red')
# ax1.set_xlim(min_date, max_date)  # Set x-axis limits

# # Create a second y-axis for hawkishness_index
# ax2 = ax1.twinx()
# ax2.plot(df['date'], df['hawkishness_index'], label='Hawkishness Index', color='blue',
#          linestyle='--')
# ax2.set_ylabel('Hawkishness Index', color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')

# # Create a third y-axis for FOMC_Sentiment_Index using the filtered BBG data
# ax3 = ax1.twinx()
# # Offset the third axis to avoid overlapping the second y-axis
# ax3.spines["right"].set_position(("outward", 60))
# ax3.plot(bbg_df_filtered['Date'], bbg_df_filtered['FOMC_Sentiment_Index'], 
#          label='FOMC Sentiment Index', color='green', linestyle='-.')
# ax3.set_ylabel('FOMC Sentiment Index', color='green')
# ax3.tick_params(axis='y', labelcolor='green')

# # Title for the plot
# plt.title('Rate Change, Hawkishness Index, and FOMC Sentiment Index Over Time')

# # Combine legends from all axes
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# lines_3, labels_3 = ax3.get_legend_handles_labels()
# lines = lines_1 + lines_2 + lines_3
# labels = labels_1 + labels_2 + labels_3
# ax1.legend(lines, labels, loc='upper left')

# plt.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load the DataFrames
# -----------------------------
# Load the BBG dataset
bbg_df = pd.read_excel(r"data\weekly_FOMC_Sentiment_data.xlsx")
print("BBG dataset:")
print(bbg_df.head())
print(bbg_df.tail())

# Load your analysis dataset
df = pd.read_csv("final_analysis.csv")
print("My analysis:")
print(df.head())
print(df.tail())

# Load the econ index dataset
econ_index_df = pd.read_csv("econ_analysis.csv")
print("econ analysis df:")
print(econ_index_df.head())
print(econ_index_df.tail())

# Rename the hawkishness_index column to econ_hawkishness_index in econ_index_df
econ_index_df.rename(columns={'hawkishness_index': 'econ_hawkishness_index'}, inplace=True)

# -----------------------------
# 2. Process Dates
# -----------------------------
# Process BBG dataset dates
bbg_df['Date'] = pd.to_datetime(bbg_df['Date'], errors='coerce')
bbg_df = bbg_df.dropna(subset=['Date'])
bbg_df = bbg_df.sort_values(by='Date').reset_index(drop=True)

# Process analysis dataset dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values(by='date').reset_index(drop=True)

# Process econ index dataset dates (assuming the date column is 'date')
econ_index_df['date'] = pd.to_datetime(econ_index_df['date'], errors='coerce')
econ_index_df = econ_index_df.sort_values(by='date').reset_index(drop=True)

# -----------------------------
# 3. Restrict Data to the Date Range in df
# -----------------------------
# Determine the date range based on the analysis dataset (rate_change)
min_date = df['date'].min()
max_date = df['date'].max()

# Filter the BBG dataframe to only include dates within the rate_change date range
bbg_df_filtered = bbg_df[(bbg_df['Date'] >= min_date) & (bbg_df['Date'] <= max_date)]

# Similarly, filter the econ index dataframe to the same date range
econ_index_df_filtered = econ_index_df[(econ_index_df['date'] >= min_date) & (econ_index_df['date'] <= max_date)]

# -----------------------------
# 4. Plot the Data
# -----------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary y-axis: Rate Change
ax1.plot(df['date'], df['rate_change'], label='Rate Change', color='red',
         linestyle='-', marker='o')
ax1.set_xlabel('Date')
ax1.set_ylabel('Rate Change', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_xlim(min_date, max_date)

# Second y-axis: Hawkishness Index from your analysis
ax2 = ax1.twinx()
ax2.plot(df['date'], df['hawkishness_index'], label='Hawkishness Index', color='blue',
         linestyle='solid')
ax2.set_ylabel('Hawkishness Index', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Third y-axis: FOMC Sentiment Index from BBG data
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
ax3.plot(bbg_df_filtered['Date'], bbg_df_filtered['FOMC_Sentiment_Index'], 
         label='FOMC Sentiment Index', color='green', linestyle='solid')
ax3.set_ylabel('FOMC Sentiment Index', color='green')
ax3.tick_params(axis='y', labelcolor='green')

# Fourth y-axis: Econ Hawkishness Index from econ_index_df
ax4 = ax1.twinx()
ax4.spines["right"].set_position(("outward", 120))
ax4.plot(econ_index_df_filtered['date'], econ_index_df_filtered['econ_hawkishness_index'], 
         label='Econ Hawkishness Index', color='purple', linestyle='solid')
ax4.set_ylabel('Econ Hawkishness Index', color='purple')
ax4.tick_params(axis='y', labelcolor='purple')

# Title for the plot
plt.title('Rate Change, Hawkishness Indices, and FOMC Sentiment Index Over Time')

# Combine legends from all axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines_3, labels_3 = ax3.get_legend_handles_labels()
lines_4, labels_4 = ax4.get_legend_handles_labels()
lines = lines_1 + lines_2 + lines_3 + lines_4
labels = labels_1 + labels_2 + labels_3 + labels_4
ax1.legend(lines, labels, loc='upper left')

plt.tight_layout()
plt.show()
