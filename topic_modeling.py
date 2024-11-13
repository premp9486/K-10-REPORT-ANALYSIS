# scripts/topic_modeling.py

import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [
        token for token in tokens if token.isalpha() and token not in stop_words
    ]
    return tokens

def topic_modeling(input_csv, output_csv, num_topics=10, num_keywords=3):
    df = pd.read_csv(input_csv)
    texts = df['Summary'].tolist()
    processed_texts = [
        preprocess(text) for text in tqdm(texts, desc="Preprocessing texts")
    ]

    # Create Dictionary and Corpus
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Build LDA model
    lda_model = models.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=15
    )

    # Extract keywords for each topic
    topics = lda_model.show_topics(
        num_topics=num_topics, num_words=num_keywords, formatted=False
    )
    topic_keywords = {
        topic_num: [word for word, _ in words] for topic_num, words in topics
    }

    topic_assignments = []
    for bow in tqdm(corpus, desc="Assigning topics"):
        topic_prob = lda_model.get_document_topics(bow)
        dominant_topic = max(topic_prob, key=lambda x: x[1])[0]
        topic_assignments.append(dominant_topic)

    df['Topic'] = topic_assignments
    df['Topic_Keywords'] = df['Topic'].apply(
        lambda x: ', '.join(topic_keywords[x])
    )
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Topic modeling completed. Results saved to {output_csv}")

if __name__ == "__main__":
    input_csv = './data/processed_data/summaries/content_summaries.csv'
    output_csv = './data/processed_data/topics/topic_modeling_results.csv'
    topic_modeling(input_csv, output_csv, num_topics=5, num_keywords=3)
