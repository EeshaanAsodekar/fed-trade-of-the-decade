import pandas as pd


def classify_stance(rate):
    if rate > 0:
        return "hawkish"
    elif rate < 0:
        return "dovish"
    else:
        return "neutral"


def load_dataset():
    df = pd.read_csv(r"C:\code\fed-trade-of-the-decade\data\processed\text_macro_market_df.csv")

    # Handle NaN values in text columns
    df['press_conference'] = df['press_conference'].fillna("")
    df['meeting_minutes'] = df['meeting_minutes'].fillna("")
    df['meeting_statements'] = df['meeting_statements'].fillna("")

    # Combine the textual columns into one
    df['combined_text'] = df['press_conference'] + " " + df['meeting_minutes'] + " " + df['meeting_statements']

    # Tag each meeting according to action taken
    df['stance'] = df['rate_change'].apply(classify_stance)

    return df


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from gensim import corpora, models
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

if __name__ == "__main__":
    df = load_dataset()
    print("\n*****************************************\n")
    print(df.head())
    print("\n*****************************************\n")
    print("\n\ndf.columns:\n",df.columns)
    print("\n*****************************************\n")
    print("\n\ndf.shape:\n",df.shape)
    print("\n*****************************************\n")
    print("\n\ndf.isna().sum():\n",df.isna().sum())

    # Ensure NLTK data is downloaded
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')

    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------
    # df = pd.read_csv('your_fomc_data.csv')  # If needed
    # Assuming df is already loaded as described

    # Filter by stance
    df_hawkish = df[df['stance'] == 'hawkish'].copy()
    df_dovish  = df[df['stance'] == 'dovish'].copy()

    # --------------------------------------------------------
    # Text Preprocessing Functions
    # --------------------------------------------------------
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        # Lowercase
        text = text.lower()
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    def tokenize(text):
        return simple_preprocess(text, deacc=True)  # removes punctuation and tokenizes

    def get_wordnet_pos(word_tag):
        """Map POS tag to first character lemmatize() accepts."""
        tag = word_tag[1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def lemmatize_tokens(tokens):
        tagged_tokens = pos_tag(tokens, lang='eng') 
        lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]
        return lemmas

    def preprocess_text(text):
        text = clean_text(text)
        tokens = tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        tokens = lemmatize_tokens(tokens)
        return tokens

    # --------------------------------------------------------
    # Preprocess the hawkish and dovish documents
    # --------------------------------------------------------
    hawkish_docs = [preprocess_text(doc) for doc in df_hawkish['combined_text'].values]
    dovish_docs  = [preprocess_text(doc) for doc in df_dovish['combined_text'].values]

    # --------------------------------------------------------
    # Create Dictionary and Corpus for Gensim
    # --------------------------------------------------------
    # For hawkish
    hawkish_dictionary = corpora.Dictionary(hawkish_docs)
    hawkish_dictionary.filter_extremes(no_below=2, no_above=0.9) # filter rare and very common terms
    hawkish_corpus = [hawkish_dictionary.doc2bow(doc) for doc in hawkish_docs]

    # For dovish
    dovish_dictionary = corpora.Dictionary(dovish_docs)
    dovish_dictionary.filter_extremes(no_below=2, no_above=0.9)
    dovish_corpus = [dovish_dictionary.doc2bow(doc) for doc in dovish_docs]

    # --------------------------------------------------------
    # Train LDA Models
    # --------------------------------------------------------
    num_topics = 5  # Adjust as needed

    hawkish_lda = models.LdaModel(
        hawkish_corpus,
        id2word=hawkish_dictionary,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    dovish_lda = models.LdaModel(
        dovish_corpus,
        id2word=dovish_dictionary,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    # --------------------------------------------------------
    # Extract Top Words from Each Topic
    # --------------------------------------------------------
    def get_top_terms_from_lda(lda_model, dictionary, topn=30):
        top_terms = set()
        for t in range(lda_model.num_topics):
            # get the top terms for each topic
            terms = lda_model.show_topic(t, topn=topn)
            for term, _ in terms:
                top_terms.add(term)
        return top_terms

    hawkish_terms = get_top_terms_from_lda(hawkish_lda, hawkish_dictionary, topn=30)
    dovish_terms  = get_top_terms_from_lda(dovish_lda, dovish_dictionary, topn=30)

    hawkish_dictionary_final = sorted(list(hawkish_terms))
    dovish_dictionary_final  = sorted(list(dovish_terms))

    print("Hawkish Dictionary (Candidate Terms):")
    print(hawkish_dictionary_final)
    print("\nDovish Dictionary (Candidate Terms):")
    print(dovish_dictionary_final)
