import os
import gdown
import tensorflow as tf

# Define the TransformerBlock class before loading the model
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Download the model from Google Drive
url = 'https://drive.google.com/uc?id=1a287f17ubldTvh5P7brg2r_VCqgVWBbB'
output = 'trained_model.h5'

# Check if the file already exists before downloading
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Ensure the file has been downloaded successfully
if not os.path.exists(output):
    raise FileNotFoundError(f"Failed to download the model file from {url}")

# Load the trained model with the custom layer
custom_objects = {'TransformerBlock': TransformerBlock}
model = tf.keras.models.load_model(output, custom_objects=custom_objects)

# The rest of your code...

# Import other necessary modules
import pandas as pd
import numpy as np
import spacy
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.impute import KNNImputer
from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf
import nltk
from ta import add_all_ta_features
from textblob import TextBlob
import re

# Load new financial news dataset
news_data = pd.read_csv('first_200_rows_dataset.csv')  # Replace with your dataset path
news_data['Date'] = pd.to_datetime(news_data['Date'])
news_data.rename(columns={'News Article': 'News_Article', 'Date': 'Date'}, inplace=True)

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Spacy model and NLTK components
nlp = spacy.load("en_core_web_sm")
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# List of companies to focus on
companies_to_focus = {
    'AMZN': 'Amazon',
    'GOOGL': 'Google',
    'AAPL': 'Apple'
}

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower()
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

# Preprocess news articles
news_data['Processed_Article'] = news_data['News_Article'].apply(preprocess_text)

# Perform Sentiment Analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

news_data["Sentiment"] = news_data["Processed_Article"].apply(get_sentiment)

# Initialize BERT tokenizer and model (You can also use RoBERTa or other advanced models)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_model = TFRobertaModel.from_pretrained('roberta-base')

def get_bert_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = model(inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use the [CLS] token's embedding

# Calculate BERT embeddings for all news
news_data["BERT_Embedding"] = news_data["Processed_Article"].apply(lambda x: get_bert_embeddings([x], tokenizer, bert_model)[0])

# Function to fetch stock prices and fundamental data for each company
def fetch_stock_prices(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.shape[0] > 14:  # Ensure there are at least 15 rows of data
            stock_data = add_all_ta_features(stock_data, open="Open", high="High", low="Low", close="Close", volume="Volume")
            # Handle missing technical indicators
            imputer = KNNImputer(n_neighbors=5)
            stock_data.iloc[:, :] = imputer.fit_transform(stock_data)
        else:
            print(f"Not enough data for {ticker}")
            return pd.DataFrame()

        # Filter out rows with missing stock prices
        stock_data.dropna(subset=['Close'], inplace=True)

        # Reset index to get the date column back after filtering
        stock_data.reset_index(inplace=True)

        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    fundamentals = stock.info
    return {
        "PE_Ratio": fundamentals.get("trailingPE", np.nan),
        "EPS": fundamentals.get("trailingEps", np.nan),
        "Revenue": fundamentals.get("totalRevenue", np.nan),
        "Market_Cap": fundamentals.get("marketCap", np.nan)
    }

# Correct date format and optionally extend the date range
from_date = "2021-01-01"
to_date = "2021-12-31"  # Extended date range

# Define look-back window
look_back = 5

# Function to prepare data for each company
def prepare_company_data(ticker, company, from_date, to_date):
    print(f"Fetching data for {company} ({ticker})")
    stock_data = fetch_stock_prices(ticker, from_date, to_date)
    if stock_data.empty:
        print(f"No stock data found for {company} ({ticker})")
        return None
    fundamental_data = fetch_fundamental_data(ticker)

    # Filter news for the company or its ticker symbol
    company_news = news_data[news_data['News_Article'].str.contains(company, case=False) | news_data['News_Article'].str.contains(ticker, case=False)]

    # Aggregate all news by day
    all_news_agg = news_data.groupby('Date').agg({
        'BERT_Embedding': lambda x: np.mean(np.vstack(x), axis=0),
        'Sentiment': 'mean'
    }).reset_index()

    # Handle missing dates for all news
    all_dates = pd.date_range(start=from_date, end=to_date, freq='D')
    all_news_agg = all_news_agg.set_index('Date').reindex(all_dates).reset_index()
    all_news_agg.rename(columns={'index': 'Date'}, inplace=True)

    # Insert neutral values for missing dates
    all_news_agg['BERT_Embedding'] = all_news_agg['BERT_Embedding'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(bert_model.config.hidden_size))
    all_news_agg['Sentiment'] = all_news_agg['Sentiment'].fillna(0.0)

    # Aggregate company-specific news by day
    if not company_news.empty:
        company_news_agg = company_news.groupby('Date').agg({
            'BERT_Embedding': lambda x: np.mean(np.vstack(x), axis=0),
            'Sentiment': 'mean'
        }).reset_index()

        # Handle missing dates for company-specific news
        company_news_agg = company_news_agg.set_index('Date').reindex(all_dates).reset_index()
        company_news_agg.rename(columns={'index': 'Date'}, inplace=True)

        # Insert neutral values for missing dates
        company_news_agg['BERT_Embedding'] = company_news_agg['BERT_Embedding'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(bert_model.config.hidden_size))
        company_news_agg['Sentiment'] = company_news_agg['Sentiment'].fillna(0.0)
    else:
        # Create empty DataFrame with the same structure
        company_news_agg = pd.DataFrame({
            'Date': all_dates,
            'BERT_Embedding': [np.zeros(bert_model.config.hidden_size)] * len(all_dates),
            'Sentiment': [0.0] * len(all_dates)
        })

    # Ensure the columns have correct suffixes
    company_news_agg.rename(columns={'BERT_Embedding': 'BERT_Embedding_company', 'Sentiment': 'Sentiment_company'}, inplace=True)
    all_news_agg.rename(columns={'BERT_Embedding': 'BERT_Embedding_all', 'Sentiment': 'Sentiment_all'}, inplace=True)

    # Merge stock data with aggregated news data
    data = pd.merge(stock_data, company_news_agg, on="Date", how="left")
    data = pd.merge(data, all_news_agg, on="Date", how="left")

    # Add fundamental data (same value for all rows as an example)
    for key, value in fundamental_data.items():
        data[key] = value

    data["Company_Name"] = company

    # Add future price column
    data["Future_Price"] = data["Close"].shift(-1)  # Shift price for prediction

    # Drop rows where the future price is missing (typically the last row)
    data.dropna(subset=['Future_Price'], inplace=True)

    # Impute missing values in technical indicators and fundamentals
    technical_indicator_columns = data.filter(like='ta_').columns
    for column in technical_indicator_columns:
        data[column].fillna(method='ffill', inplace=True)
        data[column].fillna(method='bfill', inplace=True)

    fundamental_columns = ["PE_Ratio", "EPS", "Revenue", "Market_Cap"]
    for column in fundamental_columns:
        data[column].fillna(method='ffill', inplace=True)
        data[column].fillna(method='bfill', inplace=True)

    return data

# Prepare data for each company
all_company_data = {ticker: prepare_company_data(ticker, company, from_date, to_date) for ticker, company in companies_to_focus.items()}

# Check for and remove any None entries
all_company_data = {ticker: data for ticker, data in all_company_data.items() if data is not None}

if not all_company_data:
    raise ValueError("No data available for any company in the specified date range.")

# Create sequences for each company
def create_sequences(data, look_back):
    sequences = []
    targets = []
    for i in range(len(data) - look_back):
        sequence = {
            "news_embeddings_company": np.stack(data["BERT_Embedding_company"].values[i:i+look_back]),
            "news_embeddings_all": np.stack(data["BERT_Embedding_all"].values[i:i+look_back]),
            "price": data["Close"].values[i:i+look_back].reshape(-1, 1),
            "sentiment_company": data["Sentiment_company"].values[i:i+look_back].reshape(-1, 1),
            "sentiment_all": data["Sentiment_all"].values[i:i+look_back].reshape(-1, 1),
            "technical_indicators": data.filter(like='ta_').values[i:i+look_back],
            "fundamentals": data[["PE_Ratio", "EPS", "Revenue", "Market_Cap"]].values[i:i+look_back]
        }
        sequences.append(sequence)
        targets.append(data["Future_Price"].values[i + look_back])  # Correctly assign the future price as target
    return sequences, np.array(targets)

company_sequences = {ticker: create_sequences(data, look_back) for ticker, data in all_company_data.items()}

# Ensure consistency of lengths
min_length = min(len(sequences) for sequences, _ in company_sequences.values())
company_sequences = {ticker: (sequences[:min_length], targets[:min_length]) for ticker, (sequences, targets) in company_sequences.items()}

# Convert sequences to arrays for model input
def convert_sequences(sequences):
    news_embeddings_company = np.array([seq["news_embeddings_company"] for seq in sequences])
    news_embeddings_all = np.array([seq["news_embeddings_all"] for seq in sequences])
    price = np.array([seq["price"] for seq in sequences])
    sentiment_company = np.array([seq["sentiment_company"] for seq in sequences])
    sentiment_all = np.array([seq["sentiment_all"] for seq in sequences])
    technical_indicators = np.array([seq["technical_indicators"] for seq in sequences])
    fundamentals = np.array([seq["fundamentals"] for seq in sequences])
    return news_embeddings_company, news_embeddings_all, price, sentiment_company, sentiment_all, technical_indicators, fundamentals

company_features = {ticker: (convert_sequences(sequences), targets) for ticker, (sequences, targets) in company_sequences.items()}

# Validate lengths of the features
for key, (value, targets) in company_features.items():
    print(f"{key} lengths: {[len(x) for x in value]}, targets length: {len(targets)}")
# Combine all features into a single array
def combine_features(features):
    combined = np.concatenate([features[0],
                               features[1],
                               features[2],
                               features[3],
                               features[4],
                               features[5],
                               features[6]], axis=-1)
    return combined

combined_features = {ticker: combine_features(features) for ticker, (features, _) in company_features.items()}
combined_features_array = np.concatenate(list(combined_features.values()), axis=0)

# Concatenate all targets into a single array along the correct axis
targets_array = np.concatenate([targets.reshape(-1, 1) for _, targets in company_features.values()], axis=0)

# Ensure the shape of targets matches the expected dimensions
targets_array = targets_array.reshape(-1, len(companies_to_focus))

# Convert targets to a DataFrame for multi-output regression
targets_df = pd.DataFrame(targets_array, columns=companies_to_focus.keys())

# Scale features
scaler = StandardScaler()
combined_features_array_scaled = scaler.fit_transform(combined_features_array.reshape(-1, combined_features_array.shape[-1]))
combined_features_array_scaled = combined_features_array_scaled.reshape(combined_features_array.shape)

# Scale the targets (future prices) individually for each company
target_scalers = {ticker: StandardScaler() for ticker in companies_to_focus.keys()}
targets_array_scaled = np.zeros_like(targets_array)

for i, ticker in enumerate(companies_to_focus.keys()):
    targets_array_scaled[:, i] = target_scalers[ticker].fit_transform(targets_array[:, i].reshape(-1, 1)).flatten()

# Convert targets to a DataFrame for multi-output regression
targets_df_scaled = pd.DataFrame(targets_array_scaled, columns=companies_to_focus.keys())

# Ensure the number of samples is the same
if combined_features_array.shape[0] != targets_df_scaled.shape[0]:
    min_samples = min(combined_features_array.shape[0], targets_df_scaled.shape[0])
    combined_features_array = combined_features_array[:min_samples]
    targets_df_scaled = targets_df_scaled.iloc[:min_samples]

# Prepare your data
tscv = TimeSeriesSplit(n_splits=5)
for train_index, val_index in tscv.split(combined_features_array):
    X_train, X_val = combined_features_array[train_index], combined_features_array[val_index]
    y_train, y_val = targets_df_scaled.values[train_index], targets_df_scaled.values[val_index]

# Function to preprocess text for BERT embeddings
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower()
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

# Define the company tickers and names
companies_to_focus = {
    'AMZN': 'Amazon',
    'GOOGL': 'Google',
    'AAPL': 'Apple'
}

# Function to fetch stock prices and fundamental data for each company
def fetch_stock_prices(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = add_all_ta_features(stock_data, open="Open", high="High", low="Low", close="Close", volume="Volume")
    return stock_data

def fetch_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    fundamentals = stock.info
    return {
        "PE_Ratio": fundamentals.get("trailingPE", np.nan),
        "EPS": fundamentals.get("trailingEps", np.nan),
        "Revenue": fundamentals.get("totalRevenue", np.nan),
        "Market_Cap": fundamentals.get("marketCap", np.nan)
    }

# Function to prepare data for each company
def prepare_company_data(ticker, company, from_date, to_date):
    stock_data = fetch_stock_prices(ticker, from_date, to_date)
    fundamental_data = fetch_fundamental_data(ticker)
    return stock_data, fundamental_data

# Prepare data for each company
data_dict = {ticker: prepare_company_data(ticker, company, from_date, to_date) for ticker, company in companies_to_focus.items()}

# Streamlit app
st.title("Stock Price Prediction with BERT and Transformer")

# Select ticker
selected_ticker = st.selectbox("Select a company", list(companies_to_focus.keys()))

# Display fundamental data
st.subheader("Fundamental Data")
fundamental_data = data_dict[selected_ticker][1]
st.write(fundamental_data)

# Display stock data and prediction chart
st.subheader(f"Stock Price and Predictions for {selected_ticker}")
stock_data = data_dict[selected_ticker][0]

# Convert date column to datetime
stock_data.reset_index(inplace=True)

# Plot actual stock prices
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(stock_data['Date'], stock_data['Close'], label='Actual Prices', color='blue')

# Make predictions for the selected ticker
predicted_prices = predict_prices(todays_news, look_back, bert_dim, combined_dim)
manual_prediction = predicted_prices[selected_ticker]

# Plot predicted prices
predicted_dates = stock_data['Date'].iloc[-len(manual_prediction):]
ax.plot(predicted_dates, manual_prediction, label='Predicted Prices', color='red')

# Add technical indicators (example with moving averages)
stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()
ax.plot(stock_data['Date'], stock_data['SMA50'], label='50-Day SMA', color='green')
ax.plot(stock_data['Date'], stock_data['SMA200'], label='200-Day SMA', color='purple')

# Customize chart
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Display news articles and sentiment
st.subheader("News Articles and Sentiment")
news_sentiment_data = news_data[news_data['News_Article'].str.contains(companies_to_focus[selected_ticker], case=False)]
news_sentiment_table = news_sentiment_data[['Date', 'News_Article', 'Sentiment']]

# Display news sentiment table in two columns
for index, row in news_sentiment_table.iterrows():
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Date:** {row['Date']}")
        st.write(f"**Article:** {row['News_Article']}")
    with col2:
        st.write(f"**Sentiment:** {row['Sentiment']:.2f}")

# Display all news sentiment chart
st.subheader("Overall News Sentiment Over Time")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(news_sentiment_data['Date'], news_sentiment_data['Sentiment'], label='Sentiment', color='orange')
ax.set_xlabel('Date')
ax.set_ylabel('Sentiment')
ax.grid(True)
st.pyplot(fig)
