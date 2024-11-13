# app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import torch
from transformers import pipeline
import nltk

nltk.download('punkt')

app = Flask(__name__)

# Load Data
sentiment_df = pd.read_csv(
    './data/processed_data/sentiment_analysis/sentiment_results.csv'
)
topic_df = pd.read_csv(
    './data/processed_data/topics/topic_modeling_results.csv'
)

# Load Summaries
summaries = topic_df[['Page', 'Summary', 'Topic', 'Topic_Keywords']]

# Limit summaries to 500 tokens
def limit_summary_tokens(summary, max_tokens=500):
    tokens = nltk.word_tokenize(summary)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        summary = ' '.join(tokens)
    return summary

summaries['Summary'] = summaries['Summary'].apply(
    lambda x: limit_summary_tokens(x, max_tokens=500)
)

# Initialize QA Pipeline
"""
device = 0 if torch.cuda.is_available() else -1
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=device
)
"""
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=-1
)

# Load Full Text for QA
def load_full_text(data_folder):
    texts = []
    text_files = [
        f for f in sorted(os.listdir(data_folder)) if f.endswith('.txt')
    ]
    for file_name in text_files:
        with open(
            os.path.join(data_folder, file_name), 'r', encoding='utf-8'
        ) as file:
            texts.append(file.read())
    return " ".join(texts)

full_text = load_full_text('./data/extracted_text/')

@app.route('/')
def index():
    # Load Profit and Loss items for dynamic rendering
    df_pl = pd.read_excel('./data/financial_statements.xlsx', sheet_name='Profit and Loss')
    profit_loss_items = df_pl['Items'].tolist()

    return render_template('index.html', profit_loss_items=profit_loss_items)

@app.route('/sentiment')
def sentiment():
    sentiment_counts = sentiment_df['Final_Sentiment'].value_counts().to_dict()
    return jsonify(sentiment_counts)

@app.route('/summary')
def summary():
    topics = summaries['Topic'].unique()
    summary_data = []
    for topic in topics:
        topic_summaries = summaries[summaries['Topic'] == topic]
        topic_keywords = topic_summaries.iloc[0]['Topic_Keywords']
        # Limit summaries to first 2 per topic for brevity
        topic_summaries_list = topic_summaries['Summary'].tolist()[:2]
        summary_data.append({
            'Topic': topic_keywords.title(),
            'Summaries': topic_summaries_list
        })
    return jsonify(summary_data)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({'answer': "Please provide a valid question."})

    # Perform QA
    try:
        answer = qa_pipeline(question=question, context=full_text)
        return jsonify({'answer': answer['answer']})
    except Exception as e:
        return jsonify({
            'answer': "I'm sorry, I couldn't find an answer to your question."
        })

if __name__ == '__main__':
    app.run(debug=True)
