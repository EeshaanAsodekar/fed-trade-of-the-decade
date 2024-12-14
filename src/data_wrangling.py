import pandas as pd

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

# Run the function
if __name__ == "__main__":
    rate_moves_df = get_rate_moves_by_meeting()
    rate_moves_df.to_csv("rate_moves.csv", index=False)
