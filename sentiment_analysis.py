import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os
from tqdm import tqdm

nltk.download('punkt')
nltk.download('vader_lexicon')

def analyze_sentiment_finbert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    labels = ['Negative', 'Neutral', 'Positive']
    return labels[predicted_class], confidence.item()

def analyze_sentiment_vader(text, vader_analyzer):
    scores = vader_analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive', scores['compound']
    elif scores['compound'] <= -0.05:
        return 'Negative', scores['compound']
    else:
        return 'Neutral', scores['compound']

def sentiment_analysis(data_folder, output_csv):
    # Initialize models with progress tracking
    with tqdm(total=2, desc="Loading models") as pbar:
        finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        finbert_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        pbar.update(1)
        vader_analyzer = SentimentIntensityAnalyzer()
        pbar.update(1)

    sentiment_results = []

    for file_name in tqdm(sorted(os.listdir(data_folder)), desc="Processing files"):
        if file_name.endswith('.txt'):
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
                # Analyze sentiment for the entire page text
                finbert_sentiment, finbert_conf = analyze_sentiment_finbert(text, finbert_tokenizer, finbert_model)
                vader_sentiment, vader_conf = analyze_sentiment_vader(text, vader_analyzer)
                final_sentiment = finbert_sentiment
                final_confidence = finbert_conf

                sentiment_results.append({
                    'Page': file_name,
                    'FinBERT_Sentiment': finbert_sentiment,
                    'FinBERT_Confidence': finbert_conf,
                    'VADER_Sentiment': vader_sentiment,
                    'VADER_Confidence': vader_conf,
                    'Final_Sentiment': final_sentiment,
                    'Confidence': final_confidence
                })

    sentiment_df = pd.DataFrame(sentiment_results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    sentiment_df.to_csv(output_csv, index=False)
    print(f"Sentiment analysis completed. Results saved to {output_csv}")

if __name__ == "__main__":
    data_folder = './data/extracted_text/'
    output_csv = './data/processed_data/sentiment_analysis/sentiment_results.csv'
    sentiment_analysis(data_folder, output_csv)
