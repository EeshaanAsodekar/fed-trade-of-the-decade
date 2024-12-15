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

