def main():
    import pandas as pd
    import re
    import nltk
    from nltk.corpus import stopwords
    from gensim import corpora, models
    from gensim.utils import simple_preprocess
    import spacy

    # Load the data
    df = pd.read_csv(r"C:\code\fed-trade-of-the-decade\data\processed\text_macro_market_df.csv")
    df['press_conference'] = df['press_conference'].fillna("")
    df['meeting_minutes'] = df['meeting_minutes'].fillna("")
    df['meeting_statements'] = df['meeting_statements'].fillna("")
    df['combined_text'] = df['press_conference'] + " " + df['meeting_minutes'] + " " + df['meeting_statements']

    def classify_stance(rate):
        if rate > 0:
            return "hawkish"
        elif rate < 0:
            return "dovish"
        else:
            return "neutral"

    df['stance'] = df['rate_change'].apply(classify_stance)

    # Filter documents by stance
    df_hawkish = df[df['stance'] == 'hawkish'].copy()
    df_dovish = df[df['stance'] == 'dovish'].copy()

    # Ensure NLTK stopwords are downloaded
    nltk.download('stopwords')

    # Download and load spaCy model
    try:
        spacy.cli.download("en_core_web_sm")
    except:
        pass
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    stop_words = set(stopwords.words('english'))

    # Preprocessing steps:
    # 1. Lowercase
    # 2. Keep alphabetic and numeric chars (avoid removing important terms)
    # 3. Tokenize and remove stopwords
    # 4. Lemmatize without restricting POS tags to retain key economic terms
    def clean_text(text):
        text = text.lower()
        # Allow letters and digits; remove other punctuation minimally
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def tokenize(text):
        return simple_preprocess(text, deacc=True)

    def lemmatize_all(tokens):
        doc = nlp(" ".join(tokens))
        # Lemmatize all tokens (not restricting POS), to ensure terms like "inflation" remain
        lemmas = [token.lemma_ for token in doc if len(token.lemma_) > 2 and token.lemma_ not in stop_words]
        return lemmas

    def preprocess_text(text):
        text = clean_text(text)
        tokens = tokenize(text)
        # Do not remove words based on length > 2 again here since we do it in lemmatize_all
        # Just remove stopwords here
        tokens = [t for t in tokens if t not in stop_words]
        tokens = lemmatize_all(tokens)
        return tokens

    hawkish_docs = [preprocess_text(doc) for doc in df_hawkish['combined_text'].values if isinstance(doc, str)]
    dovish_docs = [preprocess_text(doc) for doc in df_dovish['combined_text'].values if isinstance(doc, str)]

    # Create dictionaries and corpora with minimal filtering
    # This will keep a larger vocabulary, increasing the chance of seeing "inflation" or related terms.
    hawkish_dictionary = corpora.Dictionary(hawkish_docs)
    # No frequency filtering here to ensure we don't lose rare but important terms.
    hawkish_corpus = [hawkish_dictionary.doc2bow(doc) for doc in hawkish_docs]

    dovish_dictionary = corpora.Dictionary(dovish_docs)
    dovish_corpus = [dovish_dictionary.doc2bow(doc) for doc in dovish_docs]

    # Increase number of topics and passes to capture more nuanced topics
    num_topics = 10
    passes = 20

    hawkish_lda = models.LdaModel(
        hawkish_corpus,
        id2word=hawkish_dictionary,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=passes,
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
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )

    def get_top_terms_from_lda(lda_model, topn=50):
        top_terms = set()
        for t in range(lda_model.num_topics):
            terms = lda_model.show_topic(t, topn=topn)
            for term, _ in terms:
                top_terms.add(term)
        return top_terms

    # Increase the number of top words per topic to 100 to capture more potential terms
    hawkish_terms = get_top_terms_from_lda(hawkish_lda, topn=100)
    dovish_terms = get_top_terms_from_lda(dovish_lda, topn=100)

    hawkish_dictionary_final = sorted(list(hawkish_terms))
    dovish_dictionary_final = sorted(list(dovish_terms))

    print("Hawkish Dictionary (Candidate Terms):")
    print(hawkish_dictionary_final)
    print("\nDovish Dictionary (Candidate Terms):")
    print(dovish_dictionary_final)

    rem = [word for word in hawkish_dictionary_final if word not in dovish_dictionary_final]
    print(rem)
    
if __name__ == "__main__":
    main()
