# scripts/prepare_visualizations.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

nltk.download('punkt')

def generate_financial_term_frequency(text_folder, output_path):
    # Define financial terms to look for
    financial_terms = [
        'revenue', 'profit', 'growth', 'investment', 'asset', 'liability',
        'equity', 'debt', 'cash', 'income'
                        ]
    term_counts = Counter()

    text_files = [f for f in sorted(os.listdir(text_folder)) if f.endswith('.txt')]

    for file_name in text_files:
        file_path = os.path.join(text_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().lower()
            tokens = word_tokenize(text)
            for term in financial_terms:
                term_counts[term] += tokens.count(term)

    terms = list(term_counts.keys())
    counts = list(term_counts.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=terms, y=counts, palette='Blues_d')
    plt.title('Frequency of Financial Terms')
    plt.xlabel('Financial Terms')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Financial term frequency chart saved to {output_path}")

def generate_word_cloud(text_folder, output_path):
    text_files = [f for f in sorted(os.listdir(text_folder)) if f.endswith('.txt')]
    full_text = ''
    for file_name in text_files:
        file_path = os.path.join(text_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            full_text += file.read().lower() + ' '

    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=set(nltk.corpus.stopwords.words('english')),
                          collocations=False).generate(full_text)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path)
    plt.close()
    print(f"Word cloud saved to {output_path}")

def generate_sentiment_distribution(sentiment_csv, output_path):
    df = pd.read_csv(sentiment_csv)
    sentiment_counts = df['Final_Sentiment'].value_counts()

    # Ensure positive is green and negative is red
    colors = {'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'}
    color_list = [colors.get(sentiment, '#6c757d') for sentiment in sentiment_counts.index]

    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, colors=color_list, autopct='%1.1f%%', startangle=140)
    plt.title('Overall Sentiment Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Sentiment distribution chart saved to {output_path}")

def generate_topic_distribution(topic_csv, output_path):
    df = pd.read_csv(topic_csv)
    topic_counts = df['Topic'].value_counts().sort_index()
    keywords = df.drop_duplicates('Topic').sort_values('Topic')['Topic_Keywords'].values

    plt.figure(figsize=(12, 6))
    sns.barplot(x=keywords, y=topic_counts.values, palette='magma')
    plt.title('Topic Distribution with Keywords')
    plt.xlabel('Topics (Top Keywords)')
    plt.ylabel('Number of Summaries')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Topic distribution chart saved to {output_path}")

def generate_sentiment_over_topics(sentiment_csv, topic_csv, output_path):
    sentiment_df = pd.read_csv(sentiment_csv)
    topic_df = pd.read_csv(topic_csv)

    # Merge on 'Chunk' or 'Page' depending on your data
    merged_df = pd.merge(topic_df, sentiment_df, left_on='Page', right_on='Page')

    # Map sentiments to numerical values
    sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    merged_df['Sentiment_Score'] = merged_df['Final_Sentiment'].map(sentiment_mapping)

    # Calculate average sentiment score per topic
    avg_sentiment = merged_df.groupby('Topic_Keywords')['Sentiment_Score'].mean().reset_index()

    # Limit to top 5 topics
    avg_sentiment = avg_sentiment.nlargest(5, 'Sentiment_Score')

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Topic_Keywords', y='Sentiment_Score', data=avg_sentiment, palette='coolwarm')
    plt.title('Average Sentiment Score per Topic')
    plt.xlabel('Topics (Top Keywords)')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Sentiment over topics chart saved to {output_path}")

def prepare_visualizations():
    sentiment_csv = './data/processed_data/sentiment_analysis/sentiment_results.csv'
    topic_csv = './data/processed_data/topics/topic_modeling_results.csv'
    images_folder = './static/images/'
    # Create the images folder if it doesn't exist
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    generate_sentiment_distribution(sentiment_csv, os.path.join(images_folder, 'sentiment_distribution.png'))
    generate_topic_distribution(topic_csv, os.path.join(images_folder, 'topic_distribution.png'))
    generate_financial_term_frequency('./data/extracted_text/', os.path.join(images_folder, 'financial_terms_frequency.png'))
    generate_word_cloud('./data/extracted_text/', os.path.join(images_folder, 'word_cloud.png'))
    generate_sentiment_over_topics(sentiment_csv, topic_csv, os.path.join(images_folder, 'sentiment_over_topics.png'))

if __name__ == "__main__":
    prepare_visualizations()
