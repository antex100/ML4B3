import pandas as pd
import numpy as np
import spacy
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.impute import KNNImputer
from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf
import keras
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.models import load_model
import nltk
from ta import add_all_ta_features
from textblob import TextBlob
import re
import os
import gdown

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

# Define the company tickers and names
companies_to_focus = {
    'AMZN': 'Amazon',
    'GOOGL': 'Google',
    'AAPL': 'Apple'
}

# Initialize tokenizer and BERT model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_model = TFRobertaModel.from_pretrained('roberta-base')

# Define lookback window
look_back = 5

# Register the custom layer for deserialization
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

# Function to preprocess text for BERT embeddings
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower().strip()
    tokens = text.split()
    return ' '.join(tokens)

# Function to get BERT embeddings
def get_bert_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = model(inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use the [CLS] token's embedding

# Function to predict future prices
def predict_prices(news_headlines, look_back_window, bert_dim, combined_dim):
    processed_articles = [preprocess_text(article) for article in news_headlines]
    bert_embeddings = [get_bert_embeddings([article], tokenizer, bert_model)[0] for article in processed_articles]

    # Ensure the embeddings have the correct shape
    bert_embeddings = bert_embeddings[-look_back_window:]
    if len(bert_embeddings) < look_back_window:
        # Pad the embeddings if there are not enough look-back days
        padding = [np.zeros((bert_dim,)) for _ in range(look_back_window - len(bert_embeddings))]
        bert_embeddings = padding + bert_embeddings

    if combined_dim > bert_dim:
        # Combine with dummy data to match the expected combined dimension
        dummy_data = np.zeros((look_back_window, combined_dim - bert_dim))
        combined_features = np.concatenate([bert_embeddings, dummy_data], axis=-1)
    else:
        combined_features = np.array(bert_embeddings)

    # Reshape for model input
    combined_features = np.array(combined_features).reshape(1, look_back_window, -1)

    # Predict using the loaded model
    predictions = model.predict(combined_features)
    return predictions

# Function to perform sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Function to fetch fundamental data for a company
def fetch_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    fundamentals = stock.info
    return {
        "PE_Ratio": fundamentals.get("trailingPE", np.nan),
        "EPS": fundamentals.get("trailingEps", np.nan),
        "Revenue": fundamentals.get("totalRevenue", np.nan),
        "Market_Cap": fundamentals.get("marketCap", np.nan)
    }

# Load the dataset
news_data = pd.read_csv('modified_first_200_rows_dataset.csv')
news_data['Date'] = pd.to_datetime(news_data['Date'])
news_data['Processed_Article'] = news_data['News_Article'].apply(preprocess_text)
news_data['Sentiment'] = news_data['Processed_Article'].apply(get_sentiment)

# Streamlit App Layout
st.title("Stock Price Prediction App")

# Fetch data
today = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=look_back * 2)).strftime('%Y-%m-%d')
end_date = today

# Get today's news headlines
todays_news = news_data[news_data['Date'] == today]

# Define dimensions
bert_dim = bert_model.config.hidden_size  # typically 768 for BERT models
combined_dim = 1543  # Update this to the correct combined dimension

# Get stock data and predictions
stock_data_dict = {}
fundamental_data_dict = {}
for ticker in companies_to_focus:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate moving averages
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
    
    stock_data_dict[ticker] = stock_data
    fundamental_data_dict[ticker] = fetch_fundamental_data(ticker)

# Call predict_prices once
news_headlines = todays_news['Processed_Article'].tolist()
predictions = predict_prices(news_headlines, look_back, bert_dim, combined_dim)
predictions_dict = {ticker: predictions[ticker] for ticker in companies_to_focus.keys()}

# Display predicted prices
st.subheader("Predicted Prices for Tomorrow")
for ticker, company in companies_to_focus.items():
    today_price = stock_data_dict[ticker]['Close'].values[-1]
    predicted_price = predictions_dict[ticker][0][0]  # Correct indexing to match prediction structure
    arrow = "⬆️" if predicted_price > today_price else "⬇️"
    color = "green" if predicted_price > today_price else "red"
    st.markdown(f"**{company} ({ticker}):** {predicted_price:.2f} {arrow}", unsafe_allow_html=True)

# Display news headlines with sentiment in a table
st.subheader("Latest News")
news_table = todays_news[['News_Article', 'Sentiment']].copy()
news_table['Sentiment_Color'] = news_table['Sentiment'].apply(lambda x: 'green' if x > 0 else 'red' if x < 0 else 'gray')
news_table['Sentiment_Display'] = news_table.apply(lambda row: f"<span style='color:{row['Sentiment_Color']}'>{row['Sentiment']:.2f}</span>", axis=1)
news_table_display = news_table[['News_Article', 'Sentiment_Display']].rename(columns={"News_Article": "News Article", "Sentiment_Display": "Sentiment"})
st.write(news_table_display.to_html(escape=False, index=False), unsafe_allow_html=True)

# Manual prediction input
st.subheader("Manual Prediction")
manual_input = st.text_input("Enter news headline for manual prediction")
manual_look_back = st.slider("Look Back Window", min_value=1, max_value=30, value=look_back)

if manual_input:
    manual_processed_article = preprocess_text(manual_input)
    manual_prediction = predict_prices([manual_processed_article], manual_look_back, bert_dim, combined_dim)

    # Display the manual prediction for each company
    for ticker, company in companies_to_focus.items():
        st.write(f"Predicted price for {company} ({ticker}): {manual_prediction[0][0]:.2f}")

# Display stock price charts with predicted prices and technical indicators
st.subheader("Stock Price Charts")
for ticker, company in companies_to_focus.items():
    stock_data = stock_data_dict[ticker]
    predicted_price = predictions_dict[ticker][0][0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], mode='lines', name='50-Day MA'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA200'], mode='lines', name='200-Day MA'))
    fig.add_trace(go.Scatter(x=[stock_data.index[-1] + timedelta(days=1)], y=[predicted_price], mode='markers', name='Predicted Price', marker=dict(color='red', size=10)))
    fig.update_layout(title=f'{company} ({ticker}) Stock Prices', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Display fundamental data below the stock prices
st.subheader("Fundamental Data")
for ticker, company in companies_to_focus.items():
    st.write(f"**{company} ({ticker})**")
    fundamentals = fundamental_data_dict[ticker]
    for key, value in fundamentals.items():
        st.write(f"{key}: {value}")
