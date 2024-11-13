```markdown
# Reliance Annual Report Analysis

## Overview

This project provides a comprehensive analysis of the "Reliance Industries Limited (RIL) Integrated Annual Report 2023-24" and previous years. Leveraging advanced Natural Language Processing (NLP) techniques and data visualization tools, it extracts, analyzes, and presents key financial metrics, sentiment insights, content summaries, and interactive visualizations. Additionally, the project includes a predictive analysis to forecast future profits using machine learning models. The final output is an intuitive web application that allows users to explore the report's data and interact with a smart chatbot for detailed inquiries.

---

## Project Structure

```plaintext
project_root/
├── app.py                         # Main Flask application
├── requirements.txt               # Required Python packages
├── requirements_2.txt             # Additional packages for prediction notebook
├── training_prediction_model.ipynb # Jupyter notebook for profit prediction
├── data/
│   ├── extracted_text/            # Extracted text files from PDF
│   ├── archive/                   # Archived annual reports from previous years
│   ├── reliance_financials.csv    # CSV file with financial data
│   ├── financial_statements.xlsx  # Excel file with financial statements
│   └── processed_data/
│       ├── sentiment_analysis/
│       │   └── sentiment_results.csv        # Sentiment analysis results
│       ├── summaries/
│       │   └── content_summaries.csv        # Summarized content
│       └── topics/
│           └── topic_modeling_results.csv   # Topic modeling results
├── scripts/
│   ├── extract_text.py            # Script to extract text from PDFs
│   ├── sentiment_analysis.py      # Script for sentiment analysis
│   ├── summarize_content.py       # Script to summarize content
│   ├── topic_modeling.py          # Script for topic modeling
│   └── prepare_visualizations.py  # Script to generate visualizations
├── static/
│   ├── css/
│   │   └── styles.css             # Custom CSS for styling
│   ├── js/
│   │   └── scripts.js             # JavaScript for interactivity
│   └── images/
│       ├── sentiment_distribution.png      # Sentiment distribution chart
│       ├── topic_distribution.png          # Topic distribution chart
│       ├── sentiment_over_topics.png       # Sentiment over topics chart
│       ├── financial_terms_frequency.png   # Financial terms frequency chart
│       ├── word_cloud.png                  # Word cloud visualization
│       ├── profit_prediction.png           # Profit prediction chart
│       └── mse_comparison.png              # Model performance comparison chart
├── templates/
│   └── index.html                 # HTML template for the web interface
└── README.md                      # Project documentation
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project_root
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages

Ensure you have **Python 3.10** or higher installed. Install the necessary Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Data Preparation

#### 4.1. Place the Annual Report PDFs

- Ensure that the **Reliance Integrated Annual Report PDFs** are placed in the `data` directory.
- Previous years' reports should be in `data/archive/`.

#### 4.2. Extract Text from PDFs

Run the `extract_text.py` script to extract text from the PDF pages:

```bash
python scripts/extract_text.py
```

This will generate aggregated text chunks in `data/extracted_text/`.

#### 4.3. Perform Sentiment Analysis

Run the `sentiment_analysis.py` script to analyze sentiment:

```bash
python scripts/sentiment_analysis.py
```

This will generate sentiment analysis results in `data/processed_data/sentiment_analysis/sentiment_results.csv`.

#### 4.4. Summarize Content

Run the `summarize_content.py` script to summarize the extracted text:

```bash
python scripts/summarize_content.py
```

This will generate summarized content in `data/processed_data/summaries/content_summaries.csv`.

#### 4.5. Perform Topic Modeling

Run the `topic_modeling.py` script to identify topics within the summaries:

```bash
python scripts/topic_modeling.py
```

This will generate topic modeling results in `data/processed_data/topics/topic_modeling_results.csv`.

#### 4.6. Generate Visualizations

Run the `prepare_visualizations.py` script to create visual representations of the analysis:

```bash
python scripts/prepare_visualizations.py
```

This will generate various charts and images in `static/images/`.

### 5. Run the Web Application

Start the Flask application:

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your web browser to access the application.

### 6. Running the Profit Prediction Notebook

Due to potential package conflicts, it is recommended to run `training_prediction_model.ipynb` in a separate conda environment.

#### 6.1. Create a New Conda Environment

```bash
conda create -n reliance_prediction_env python=3.10
conda activate reliance_prediction_env
```

#### 6.2. Install Required Packages

Install the necessary packages using `requirements_2.txt`:

```bash
pip install -r requirements_2.txt
```

#### 6.3. Run the Notebook

```bash
jupyter notebook training_prediction_model.ipynb
```

This notebook will perform the profit prediction modeling and generate additional visualizations saved in `static/images/`.

---

## Detailed Analysis and Methodology

### 1. Data Extraction and Preparation

#### OCR Methods

**Optical Character Recognition (OCR)** is a technology that converts different types of documents, such as scanned paper documents, PDFs, or images captured by a digital camera, into editable and searchable data. In this project, OCR techniques were employed to extract text from the Reliance Annual Report PDFs.

- **Text Extraction**: Utilizing the `PyPDF2` library, text content was extracted from each page of the PDFs. To maintain context and improve analysis accuracy, every five pages were aggregated into a single text chunk.

- **Table Extraction**: Financial tables from the reports were manually organized into `reliance_financials.csv` and `financial_statements.xlsx` files. This structured approach facilitates accurate financial analysis and visualization.

### 2. Sentiment Analysis

#### Tools and Models

- **FinBERT (`from transformers import AutoTokenizer, AutoModelForSequenceClassification`)**:

  - **Purpose**: FinBERT is a BERT-based model fine-tuned specifically for financial text sentiment analysis. It classifies text into Positive, Neutral, or Negative sentiments with high accuracy in financial contexts.

  - **Reason for Use**: Financial reports contain domain-specific language and nuances. FinBERT provides more precise sentiment classifications tailored to the financial industry compared to general-purpose sentiment models.

- **VADER Sentiment Analyzer (`from nltk.sentiment.vader import SentimentIntensityAnalyzer`)**:

  - **Purpose**: VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analysis tool that excels at analyzing social media texts.

  - **Reason for Use**: VADER offers a complementary sentiment analysis perspective, ensuring robustness by capturing overall sentiment intensity alongside FinBERT's domain-specific classifications.

#### Process

1. **Text Tokenization**: Each aggregated text chunk was tokenized into sentences using `nltk.tokenize.sent_tokenize`.

2. **Sentiment Scoring**:

   - **FinBERT**: Each sentence was analyzed to determine its sentiment (Positive, Neutral, Negative) along with a confidence score.

   - **VADER**: Each sentence was assigned a sentiment score based on its compound score, categorizing it similarly.

3. **Final Sentiment Determination**: The final sentiment for each text chunk was determined primarily based on FinBERT's classification and confidence score, ensuring high-confidence sentiment classification.

4. **Output**: Sentiment analysis results were stored in `data/processed_data/sentiment_analysis/sentiment_results.csv`.

### 3. Summarization

#### Model Used

- **BART (`from transformers import pipeline`)**:

  - **Purpose**: BART (Bidirectional and Auto-Regressive Transformers) is a transformer model designed for sequence-to-sequence tasks, including text summarization.

  - **Reason for Use**: BART effectively generates coherent and concise summaries by understanding the context and structure of the input text, making it ideal for summarizing lengthy financial reports.

#### Process

1. **Text Preparation**: Each aggregated text chunk was prepared for summarization, ensuring it did not exceed 2000 words to fit the model's input constraints.

2. **Summarization**: BART generated a concise summary for each text chunk, which was then limited to 500 tokens to maintain brevity and relevance.

3. **Output**: Summarized content was stored in `data/processed_data/summaries/content_summaries.csv`.

### 4. Topic Modeling

#### Tools and Libraries

- **Gensim (`from gensim import corpora, models`)**:

  - **Purpose**: Gensim is a robust library for topic modeling and document similarity analysis. It provides efficient implementations of algorithms like LDA (Latent Dirichlet Allocation).

  - **Reason for Use**: Gensim's LDA model is effective in uncovering hidden thematic structures within large corpora of text, making it suitable for identifying topics in the annual report summaries.

- **NLTK (`from nltk.tokenize import word_tokenize`)**:

  - **Purpose**: NLTK (Natural Language Toolkit) offers tools for text preprocessing, including tokenization and stopword removal.

  - **Reason for Use**: Effective text preprocessing is crucial for accurate topic modeling. NLTK provides the necessary tools to clean and prepare text data for analysis.

#### Process

1. **Preprocessing**: Summarized content was tokenized and cleaned by removing stopwords and non-alphabetic tokens using NLTK.

2. **Dictionary and Corpus Creation**: A dictionary mapping of words to unique IDs was created, and a corpus representing each document as a bag-of-words was generated.

3. **LDA Model Training**: The LDA model was trained to identify 10 topics within the summaries, each characterized by the top 3 keywords.

4. **Topic Assignment**: Each summary was assigned to the dominant topic based on the highest probability score from the LDA model.

5. **Output**: Topic modeling results were stored in `data/processed_data/topics/topic_modeling_results.csv`, including topic assignments and keywords for each summary.

### 5. Profit Prediction Modeling

#### Overview

The project includes a predictive analysis aimed at forecasting the "Profit for the Year" for Reliance Industries Limited. By utilizing historical annual reports and financial data, the model predicts the profit for the next financial year (N+1) based on the data from the current year (N).

#### Procedure in `training_prediction_model.ipynb`

1. **Data Collection**:

   - **Text Extraction**: Text is extracted from each page of the annual reports from previous years (2018-19 to 2023-24) using OCR methods and saved in the `data/archive/extracted_text/` directory.

   - **Financial Data**: The `reliance_financials.csv` file contains historical profit data.

2. **Text Vectorization using SpaCy**:

   - **SpaCy Library**: An advanced NLP library that provides powerful tools for text preprocessing and vectorization. SpaCy's pre-trained language models efficiently convert text into numerical vectors that capture semantic information.

   - **Vectorization**: Each page's text is converted into a numerical vector using SpaCy's `en_core_web_md` model. This process transforms unstructured text into structured numerical data suitable for machine learning models.

3. **Feature Engineering**:

   - **Combining Vectors and Profit Information**: For each page in year **N**, the feature vector includes the page's text vector and the profit of year **N**. This combination allows the model to consider both textual content and financial performance.

   - **Target Variable**: The profit for year **N+1** is used as the target variable for each sample corresponding to year **N**.

4. **Model Training**:

   - **Models Used**:

     - **Linear Regression**: Provides a simple baseline to assess linear relationships between features and the target.

     - **Neural Network**: Captures non-linear relationships through multiple layers and activation functions.

     - **LSTM (Long Short-Term Memory)**: A type of recurrent neural network effective for sequential data, potentially capturing temporal dependencies.

   - **Training Process**:

     - The models are trained on data from years **2018-19** to **2021-22**.

     - The test data is from **2022-23**.

     - **Evaluation Metric**: Mean Squared Error (MSE) is used to evaluate model performance on the test set.

5. **Prediction**:

   - The best-performing model (based on MSE) is used to predict the profit for **2024-25** using data from **2023-24**.

6. **Results Visualization**:

   - **Profit Prediction Chart**: Displays historical profits and the predicted profit for **2024-25**.

   - **Model Performance Comparison**: A bar chart comparing the MSE of the models to visualize their performance.

#### Interest of Using Different Methods

- **Comparative Analysis**: Testing multiple models allows for identifying the most effective approach for the data, ensuring robustness in predictions.

- **Linear Regression**: Offers simplicity and interpretability, serving as a benchmark for more complex models.

- **Neural Networks**: Capable of modeling complex non-linear relationships between features and the target variable.

- **LSTM Networks**: Designed to handle sequential data and capture temporal dependencies, potentially improving predictions based on the sequence of annual reports.

#### Using N to Predict N+1

- **Approach**: Data from year **N**, including textual content and financial profit information, is used to predict the profit for year **N+1**.

- **Benefit**: Incorporates the most recent information to forecast future performance, leveraging both qualitative (textual) and quantitative (financial) data.

- **Process**:

  - For each page in year **N**, the model uses the page's vectorized text and the profit of year **N** as input features.

  - The target variable is the profit of year **N+1**.

  - This method allows the model to learn relationships between the content of the reports and future financial performance.

---

## Deep Learning Models Used

### SpaCy for Text Vectorization

- **Library**: `spacy`

- **Purpose**: SpaCy provides advanced NLP capabilities, including tokenization, part-of-speech tagging, and vectorization.

- **Reason for Use**: Efficiently converts large amounts of text into numerical vectors that capture semantic meaning, suitable for input into machine learning models.

### FinBERT for Sentiment Analysis

- **Library**: `from transformers import AutoTokenizer, AutoModelForSequenceClassification`

- **Purpose**: Specialized for financial text sentiment analysis.

- **Reason for Use**: Provides accurate sentiment classifications tailored to the financial domain.

### VADER Sentiment Analyzer

- **Library**: `from nltk.sentiment.vader import SentimentIntensityAnalyzer`

- **Purpose**: Rule-based tool for sentiment analysis.

- **Reason for Use**: Offers a complementary perspective to FinBERT.

### BART for Summarization

- **Library**: `from transformers import pipeline`

- **Purpose**: Generates coherent and concise summaries.

### Gensim for Topic Modeling

- **Library**: `from gensim import corpora, models`

- **Purpose**: Implements LDA for topic modeling.

### Neural Networks and LSTM for Prediction

- **Libraries**: `keras`, `tensorflow`

- **Purpose**:

  - **Neural Network**: Captures non-linear relationships in the data.

  - **LSTM**: Handles sequential data and potential temporal dependencies.

---

## Running the Application

### Ensure All Data Processing Scripts Have Been Executed

Follow the [Setup Instructions](#setup-instructions) to extract text, perform sentiment analysis, summarize content, conduct topic modeling, and generate visualizations.

### Start the Flask Application

Run the `app.py` script to launch the web application:

```bash
python app.py
```

### Access the Web Application

Open your web browser and navigate to `http://127.0.0.1:5000` to interact with the analysis dashboard and chatbot.

---

## Future Improvements

- **Enhanced Chatbot**: Integrate more sophisticated NLP models for deeper understanding and more accurate responses to user queries.

- **Containerization**: Dockerize the application for scalable and portable deployment across different environments.

- **Cloud Deployment**: Deploy the application to cloud platforms to facilitate wider accessibility and better performance.

- **Real-Time Data Processing**: Implement real-time data extraction and analysis to keep the dashboard up-to-date with the latest financial reports.

- **Expanded Predictive Modeling**: Incorporate additional financial indicators and more years of data to improve the accuracy of profit predictions.

---

## Acknowledgements

- **Reliance Industries Limited**: For providing the integrated annual report data.

- **SpaCy, NLTK, and Gensim Communities**: For providing robust tools and libraries essential for natural language processing tasks.

- **Matplotlib, Seaborn, and WordCloud**: For offering comprehensive data visualization capabilities.
