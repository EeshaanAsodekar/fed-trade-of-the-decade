import pandas as pd
import os

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

# Run the function
if __name__ == "__main__":
    rate_moves_df = get_rate_moves_by_meeting()
    rate_moves_df.to_csv("data/processed/rate_moves.csv", index=False)

    text_df = get_text_dataset()
    print(text_df.head())
    print(text_df.tail())