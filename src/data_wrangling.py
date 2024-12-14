import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to get rate moves by meeting
def get_rate_moves_by_meeting():
    # Load the FOMC meeting dates
    fomc_meeting_dates_path = "data/processed/fomc_meeting_dates.csv"
    fomc_dates_df = pd.read_csv(fomc_meeting_dates_path)
    fomc_dates_df["date"] = pd.to_datetime(fomc_dates_df["date"])
    print("FOMC Meeting Dates:")
    print(fomc_dates_df.tail())
    
    # Load the Fed funds target rate data
    fed_funds_rate_path = "data/raw/fed_funds_target_rate.csv"
    fed_funds_df = pd.read_csv(fed_funds_rate_path)
    fed_funds_df["date"] = pd.to_datetime(fed_funds_df["date"])
    print("Fed Funds Target Rate:")
    print(fed_funds_df.tail())

    # Initialize a list to store the results
    rate_changes = []

    # Iterate over each FOMC meeting date
    for i in range(len(fomc_dates_df)):
        meeting_date = fomc_dates_df.loc[i, "date"]
        print(f"Processing meeting date: {meeting_date}")

        # Get the rate on the meeting date
        rate_on_meeting_date = fed_funds_df.loc[fed_funds_df["date"] == meeting_date, "rate"]
        print(f"Rate on meeting date: {rate_on_meeting_date.values}")

        # Initialize tgt_rate as None
        tgt_rate = None

        # Check if there is a rate change on the meeting day itself
        if not rate_on_meeting_date.empty:
            rate_before_meeting = fed_funds_df.loc[fed_funds_df["date"] < meeting_date, "rate"]
            if not rate_before_meeting.empty:
                previous_rate = rate_before_meeting.iloc[-1]
                if rate_on_meeting_date.iloc[0] != previous_rate:
                    rate_change = rate_on_meeting_date.iloc[0] - previous_rate
                    tgt_rate = rate_on_meeting_date.iloc[0]
                    rate_changes.append({"date": meeting_date, "rate_change": rate_change, "tgt_rate": tgt_rate})
                    print(f"Rate changed on meeting day: {rate_change}, Target rate: {tgt_rate}")
                    continue  # Skip to the next meeting

        # Get the rate for the day after the meeting
        next_day_rates = fed_funds_df.loc[fed_funds_df["date"] > meeting_date, "rate"]
        rate_after_meeting = next_day_rates.iloc[0] if not next_day_rates.empty else None
        print(f"Rate after meeting: {rate_after_meeting}")

        # Check and calculate the rate change for the next day
        if not rate_on_meeting_date.empty and rate_after_meeting is not None:
            rate_change = rate_after_meeting - rate_on_meeting_date.iloc[0]
            tgt_rate = rate_after_meeting
            rate_changes.append({"date": meeting_date, "rate_change": rate_change, "tgt_rate": tgt_rate})
            print(f"Rate changed the next day: {rate_change}, Target rate: {tgt_rate}")
        else:
            print(f"Issue with meeting date {meeting_date}: Missing data.")

    # Create a DataFrame from the results
    rate_changes_df = pd.DataFrame(rate_changes)

    # Display the DataFrame
    print("Rate Changes:")
    print(rate_changes_df.head())

    return rate_changes_df




### creating dataframe of fed communications and fed rate moves
def get_text_dataset():
    """
    Loads the rate moves CSV and adds text columns for FOMC communications.
    The FOMC communications are pulled from the respective directories based on meeting dates.

    Returns:
        pd.DataFrame: DataFrame containing rate change, target rate, and FOMC communication texts.
    """
    # Path configurations
    rate_moves_path = "data/processed/rate_moves.csv"
    press_conf_dir = "data/raw/fomc_press_conf/texts"
    meeting_minutes_dir = "data/raw/fomc_meeting_minutes/texts"
    meeting_statements_dir = "data/raw/fomc_meeting_statements/texts"
    
    # Load rate moves data
    rate_moves_df = pd.read_csv(rate_moves_path)
    rate_moves_df["date"] = pd.to_datetime(rate_moves_df["date"])
    
    # Initialize columns for FOMC communication texts
    rate_moves_df["press_conference"] = ""
    rate_moves_df["meeting_minutes"] = ""
    rate_moves_df["meeting_statements"] = ""

    # Helper function to load text from a file
    def load_text(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    # Iterate over each row and add FOMC communication texts
    for idx, row in rate_moves_df.iterrows():
        date_str = row["date"].strftime("%Y%m%d")  # Format date to match filenames
        
        # Press conference
        press_conf_file = os.path.join(press_conf_dir, f"FOMCpresconf{date_str}.txt")
        rate_moves_df.at[idx, "press_conference"] = load_text(press_conf_file)
        
        # Meeting minutes
        meeting_minutes_file = os.path.join(meeting_minutes_dir, f"fomcminutes{date_str}.txt")
        rate_moves_df.at[idx, "meeting_minutes"] = load_text(meeting_minutes_file)
        
        # Meeting statements
        # Handles variations like `a1` in meeting statement filenames
        statement_file = os.path.join(meeting_statements_dir, f"monetary{date_str}a1.txt")
        rate_moves_df.at[idx, "meeting_statements"] = load_text(statement_file)
        rate_moves_df.to_csv("data/processed/meeting_rate_and_text_data.csv")
    return rate_moves_df





#### TODO: NEED TO CROSS VALIDATE THE PCT CHANGE IS BEING IMPLMENTED CORRECTLY!
def get_text_macro_market_dataset(df_data_path, df_text_path):
    """
    Creates a master DataFrame by adding columns to df_text that represent the changes 
    in macro and market variables from df_data between consecutive FOMC meeting dates.
    If a meeting date does not exist in df_data, the closest date (forward or backward) is used.
    The function also plots the percentage changes for all columns.

    Args:
        df_data_path (str): Path to the interpolated macro and market data CSV file.
        df_text_path (str): Path to the FOMC meeting rate and text data CSV file.

    Returns:
        pd.DataFrame: A master DataFrame with added change columns for each macro/market variable.
    """
    # Load the datasets
    df_data = pd.read_csv(df_data_path)
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    
    # Convert all columns except 'Date' to floats
    for col in df_data.columns:
        if col != 'Date':
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce')

    df_text = pd.read_csv(df_text_path)
    df_text['date'] = pd.to_datetime(df_text['date'])
    
    # Initialize a dictionary to store changes for each variable
    changes = {col: [] for col in df_data.columns if col != 'Date'}

    # Iterate over FOMC meeting dates
    for i, current_date in enumerate(df_text['date']):
        if i == 0:
            # Get the first valid row in df_data for the current_date
            initial_row = df_data[df_data['Date'] <= current_date]
            current_row = df_data[df_data['Date'] >= current_date]

            if initial_row.empty:
                initial_row = df_data.iloc[0]  # Use the first available row in df_data
            else:
                initial_row = initial_row.iloc[-1]  # Use the last row before or on the current_date

            if current_row.empty:
                current_row = df_data.iloc[-1]  # Use the last available row in df_data
            else:
                current_row = current_row.iloc[0]  # Use the first row after or on the current_date

        else:
            # Handle subsequent meeting dates
            previous_date = df_text['date'].iloc[i - 1]
            initial_row = df_data[df_data['Date'] >= previous_date]

            if initial_row.empty:
                initial_row = df_data.iloc[-1]  # Use the last available row in df_data
            else:
                initial_row = initial_row.iloc[0]  # Use the first row after or on the previous_date

            current_row = df_data[df_data['Date'] >= current_date]
            if current_row.empty:
                current_row = df_data.iloc[-1]  # Use the last available row in df_data
            else:
                current_row = current_row.iloc[0]  # Use the first row after or on the current_date
        
        # Compute changes for each column
        for col in changes:
            initial_value = initial_row[col]
            current_value = current_row[col]

            # Handle edge cases for NaN and near-zero initial values
            if pd.notna(initial_value) and initial_value != 0:
                changes[col].append((current_value - initial_value) / initial_value)
            else:
                # Assign NaN for invalid cases
                changes[col].append(None)

    # Add the computed changes as columns to df_text
    for col, change_values in changes.items():
        df_text[f"pct_change_in_{col}"] = change_values
    
    # Plot percentage changes for each column- SANTIY CHECK
    plt.figure(figsize=(12, 8))
    for col in changes.keys():
        pct_change_col = f"pct_change_in_{col}"
        if pct_change_col in df_text:
            plt.scatter(df_text['date'], df_text[pct_change_col], label=pct_change_col, s=10)  # Use scatter for points
    plt.title('Percentage Changes in Macro and Market Variables')
    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # validating the 2y 10y and 2y 10y spread - SANITY CHECK
    selected_columns = ["pct_change_in_10-Year Treasury Yield", 
                        "pct_change_in_2-Year Treasury Yield", 
                        "pct_change_in_2y-10y Spread"]
    plt.figure(figsize=(12, 6))
    for col in selected_columns:
        if col in df_text:
            plt.scatter(df_text['date'], df_text[col], label=col, s=10)  # Plot as points

    plt.title('Percentage Changes in Selected Market Variables')
    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    # Format the x-axis to show dates in mm-dd-yy format
    plt.xticks(df_text['date'], df_text['date'].dt.strftime('%m-%d-%y'), rotation=45, fontsize=8)
    plt.tight_layout()
    plt.show()

    return df_text





if __name__ == "__main__":
    rate_moves_df = get_rate_moves_by_meeting()
    rate_moves_df.to_csv("data/processed/rate_moves.csv", index=False)

    text_df = get_text_dataset()
    print(text_df.head())
    print(text_df.tail())

    master_df = get_text_macro_market_dataset(
        r"data/processed/interpolated_all_macro_market_data_2012_present.csv",
        r"data/processed/meeting_rate_and_text_data.csv"
    )
    print(master_df.head())
    master_df.to_csv("data/processed/text_macro_market_df.csv", index=False)