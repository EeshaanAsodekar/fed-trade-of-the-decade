import pandas as pd
import math
import warnings
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import math
import warnings
import pandas as pd
from collections import Counter
import matplotlib.dates as mdates


def load_dataset():
    df = pd.read_csv(r"C:\code\fed-trade-of-the-decade\data\processed\text_macro_market_df.csv")

    # Handle NaN values in text columns
    df['press_conference'] = df['press_conference'].fillna("")
    df['meeting_minutes'] = df['meeting_minutes'].fillna("")
    df['meeting_statements'] = df['meeting_statements'].fillna("")

    # Combine the textual columns into one
    df['combined_text'] = df['press_conference'] + " " + df['meeting_minutes'] + " " + df['meeting_statements']

    return df





def get_hawkish_dovish_score_from_df(dictionary_path: str, df: pd.DataFrame, text_column: str, hawk_or_dove: str) -> pd.DataFrame:
    """
    Calculate hawkish/dovish word scores for documents provided in a DataFrame.
    The DataFrame must have a text column specified by `text_column`.
    
    Args:
        dictionary_path (str): Path to the dictionary file containing hawkish/dovish words.
        df (pd.DataFrame): DataFrame containing the specified text column.
        text_column (str): Name of the column in df containing the text to analyze.
        hawk_or_dove (str): String indicating whether we're calculating 'hawkish' or 'dovish' scores.
    
    Returns:
        pd.DataFrame: The original DataFrame with a new column added containing the weighted hawkish/dovish score.
    """
    
    # Load the hawkish/dovish words from the dictionary file into a list
    with open(dictionary_path, 'r') as file:
        words_list = [line.strip().lower() for line in file.readlines()]

    # Initialize a dictionary to store hawkish/dovish word counts for each document
    word_counts = {word: [] for word in words_list}

    # List to store total word count for each document
    total_word_count = []

    # Iterate over each row in the DataFrame to count occurrences of words in the given column
    for idx, row in df.iterrows():
        text = str(row[text_column]).lower()
        tokens = text.split()
        word_counter = Counter(tokens)

        # Append total word count for the current document
        total_word_count.append(len(tokens))

        # Count occurrences of each hawkish/dovish word and store in word_counts
        for word in words_list:
            word_counts[word].append(word_counter.get(word, 0))

    # Create a DataFrame with hawkish/dovish word counts for each document
    count_matrix_df = pd.DataFrame(word_counts, index=df.index)

    # Create a DataFrame for total word counts in each document
    stats_df = pd.DataFrame({'Total_Words': total_word_count}, index=df.index)

    # Suppress FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Calculate TF-IDF for hawkish/dovish word counts
    df_tf_idf = count_matrix_df.copy()
    df_i = [0] * count_matrix_df.shape[1]
    n_docs = df_tf_idf.shape[0]

    # Compute TF
    for i in range(n_docs):  # Iterate over documents
        doc_word_count = stats_df['Total_Words'].iloc[i]
        for j in range(df_tf_idf.shape[1]):  # Iterate over words
            count = df_tf_idf.iloc[i, j]
            if count > 0:
                # TF = (1 + log(count)) / (1 + log(total words in doc))
                df_tf_idf.iloc[i, j] = (1 + math.log(count)) / (1 + math.log(doc_word_count))
                df_i[j] += 1
            else:
                df_tf_idf.iloc[i, j] = 0

    # Compute IDF
    # IDF = log(N / number_of_docs_containing_word)
    idf_values = [math.log(n_docs / i) if i else 0 for i in df_i]

    # Multiply TF by IDF
    df_tf_idf = df_tf_idf.mul(idf_values, axis=1)

    # Weighting: Multiply TF-IDF by raw word counts to calculate weighted hawkish/dovish word scores
    weighted_counts = df_tf_idf * count_matrix_df

    # Sum the weighted scores for each document
    weighted_sum = weighted_counts.sum(axis=1)

    # Create a column name for the score
    score_col_name = f"{hawk_or_dove}ish_{text_column}_score"
    
    # Add this new column to the original df
    df[score_col_name] = weighted_sum

    return df


import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
import matplotlib.pyplot as plt

def get_composite_hawkishness_index():
    df = pd.read_csv(r'test.csv')
    print(df.head())
    print(df.columns)

    # Step 1: Define the target
    # Here we use rate_change as a proxy for hawkishness measure
    y = df['rate_change']

    # Step 2: Select candidate features
    # We'll pick all hawkish scores and some macro variables.
    # Adjust the list as needed.
    candidate_features = [
        'hawkish_combined_text_score',
        'tgt_dist_PCE Inflation YoY Change',
        'tgt_dist_Unemployment Rate',
        'pct_change_in_CPI (All Urban Consumers) YoY Change',
        'pct_change_in_Core CPI (Ex Food & Energy) YoY Change',
        'pct_change_in_PCE Inflation YoY Change',
        'pct_change_in_Core PCE Inflation (Ex Food & Energy) YoY Change',
        'pct_change_in_Real GDP',
        'pct_change_in_10-Year Treasury Yield',
        'pct_change_in_2-Year Treasury Yield',
        'pct_change_in_2-10 Spread',
        'pct_change_in_Total Nonfarm Payrolls',
    ]

    X = df[candidate_features].copy()

    # Step 3: Basic Correlation Analysis (optional)
    corr_matrix = X.corrwith(y)
    print("Correlation with Target:\n", corr_matrix.sort_values(ascending=False))

    # # Drop features with very low correlation to simplify (optional)
    # low_corr_feats = corr_matrix[abs(corr_matrix)<0.05].index
    # X.drop(columns=low_corr_feats, inplace=True)

    # Step 4: Use a LassoCV to select features
    # For time series data, consider a time-series split
    tscv = TimeSeriesSplit(n_splits=5)
    lasso = LassoCV(cv=tscv, random_state=42, max_iter=50000)
    lasso.fit(X, y)

    # Print out coefficients
    feature_coef = pd.Series(lasso.coef_, index=X.columns)
    print("Lasso selected features and their coefficients:\n", feature_coef)

    # Keep only features with non-zero coefficients
    selected_features = feature_coef[feature_coef!=0].index.tolist()
    print("Selected Features:", selected_features)

    # Step 5: Fit a statsmodels OLS model with the selected features
    X_selected = sm.add_constant(X[selected_features])
    model = sm.OLS(y, X_selected).fit()
    print(model.summary())

    # Interpret the model summary:
    # - Look at p-values to further refine your set of features.
    # - Features with low p-values are more statistically significant.
    # - If needed, remove features that are not significant and re-run until you get a stable set.

    # Once stable, you have a model such as:
    # fed_hawkishness = sum_{i}(coeff_i * feature_i)

    # Step 6: Create the Hawkishness Index
    # Use the selected features and their coefficients to compute the index
    hawkishness_index = X[selected_features] @ feature_coef[selected_features]

    # Add the index to the DataFrame
    df['hawkishness_index'] = hawkishness_index

    # Display the resulting DataFrame with the Hawkishness Index
    print(df[['date', 'hawkishness_index']].head())


    import matplotlib.dates as mdates

    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Plot Lasso predictions against actual rate_change
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot actual rate_change on the primary y-axis
    ax1.plot(df['date'], y, label='Actual Rate Change', color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Rate Change', color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')

    # Add predicted hawkishness index on the second y-axis
    ax2 = ax1.twinx()
    ax2.plot(df['date'], hawkishness_index, label='Predicted Hawkishness Index', color='blue', linewidth=2)
    ax2.set_ylabel('Hawkishness Index', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')

    # Format the x-axis to show only the year
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Year format
    ax1.xaxis.set_major_locator(mdates.YearLocator())  # Tick every year

    # Reduce text size and rotate the x-axis labels for clarity
    plt.xticks(fontsize=10, rotation=0)

    # Add titles and legend
    plt.title('Lasso Model: Actual Rate Change vs Predicted Hawkishness Index', fontsize=14)
    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print(df.columns)

    score_df = get_hawkish_dovish_score_from_df(
        dictionary_path="data/processed/dictionaries/hawk_unique.txt", 
        df=df, 
        text_column="meeting_minutes", 
        hawk_or_dove="hawk"
    )

    score_df = get_hawkish_dovish_score_from_df(
        dictionary_path="data/processed/dictionaries/hawk_unique.txt", 
        df=df, 
        text_column="press_conference", 
        hawk_or_dove="hawk"
    )

    score_df = get_hawkish_dovish_score_from_df(
        dictionary_path="data/processed/dictionaries/hawk_unique.txt", 
        df=df, 
        text_column="meeting_statements", 
        hawk_or_dove="hawk"
    )

    score_df = get_hawkish_dovish_score_from_df(
        dictionary_path="data/processed/dictionaries/hawk_unique.txt", 
        df=df, 
        text_column="combined_text", 
        hawk_or_dove="hawk"
    )

    print(score_df.columns)

    score_df.to_csv("test.csv",index=False) 
    print(score_df.head())


    ##### plotting scores by meeting
    # Define the columns to plot
    hawkish_scores = [
        ('hawkish_meeting_minutes_score', 'Meeting Minutes Score'),
        ('hawkish_press_conference_score', 'Press Conference Score'),
        ('hawkish_meeting_statements_score', 'Meeting Statements Score'),
        ('hawkish_combined_text_score', 'Combined Score')
    ]

    # Plot each hawkishness score in a separate subplot
    for hawkish_column, label in hawkish_scores:
        plt.figure(figsize=(14, 6))

        # Plot hawkishness score
        ax1 = plt.gca()  # Primary axis
        ax1.plot(df['date'], df[hawkish_column], label=label, marker='o', color='blue')
        ax1.set_xlabel('Meeting Date')
        ax1.set_ylabel(label, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title(f'{label} and Rate Change by Meeting')
        ax1.grid()

        # Plot rate change on a secondary axis
        ax2 = ax1.twinx()  # Secondary axis
        ax2.plot(df['date'], df['rate_change'], label='Rate Change', color='black', linestyle='--', marker='x')
        ax2.set_ylabel('Rate Change', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add a legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    get_composite_hawkishness_index()