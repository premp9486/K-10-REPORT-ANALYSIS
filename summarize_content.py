
# summarize_content.py

from transformers import pipeline
import os
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm
import torch

nltk.download('punkt')

def summarize_text(summarizer, text, max_length=150, min_length=40):
    summaries = []
    for i in range(0, len(text), 1000):
        chunk = text[i:i+1000]
        # Set adjusted_max_length based on the chunk's length but enforce minimum constraints
        adjusted_max_length = min(max_length, len(sent_tokenize(chunk)) * 20)
        adjusted_max_length = max(adjusted_max_length, min_length + 10)  # Ensure it's above min_length
        summary = summarizer(chunk, max_length=adjusted_max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)

def summarize_content(data_folder, output_csv):
    # Select device based on GPU availability
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn', device=device)
    summaries = []

    for file_name in tqdm(sorted(os.listdir(data_folder)), desc="Processing files"):
        if file_name.endswith('.txt'):
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                summary = summarize_text(summarizer, text)
                summaries.append({
                    'Page': file_name,
                    'Summary': summary
                })

    # Ensure all operations are performed on the CPU after processing
    if torch.cuda.is_available():
        summarizer.model.to('cpu')

    summary_df = pd.DataFrame(summaries)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"Content summarization completed. Summaries saved to {output_csv}")

if __name__ == "__main__":
    data_folder = './data/extracted_text/'
    output_csv = './data/processed_data/summaries/content_summaries.csv'
    summarize_content(data_folder, output_csv)