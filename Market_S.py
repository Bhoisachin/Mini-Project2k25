import streamlit as st
st.set_page_config(page_title="MARKET.AI", layout="wide")
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date, datetime
import yfinance as yf
import google.generativeai as genai
from dotenv import load_dotenv
import time
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt
import json
from faster_whisper import WhisperModel
import subprocess
from transformers import pipeline
import re

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Initialize Whisper model
try:
    whisper_model = WhisperModel("base", compute_type="int8")
except Exception as e:
    st.error(f"Failed to initialize Whisper model: {e}")
    whisper_model = None

# Initialize stock name dictionary
@st.cache_data
def load_stock_data():
    """Load and cache stock list from CSV."""
    try:
        df = pd.read_csv('stock_list_cleaned.csv')
        return {row['Stock Name'].strip().lower(): row['Symbol'].strip() for _, row in df.iterrows()}
    except FileNotFoundError:
        st.error("stock_list_cleaned.csv not found.")
        return {}

stock_name_dict = dict(sorted(load_stock_data().items(), key=lambda item: item[0][0]))

def initialize_ai():
    """Configure Google Gemini AI."""
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.error(f"Failed to initialize AI: {e}")
        return None

def call_ai(prompt='Hello MARKET.AI'):
    """Call AI model with retry logic."""
    model = initialize_ai()
    if not model:
        return "AI service unavailable."
    for _ in range(3):  # Retry up to 3 times
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            time.sleep(1)
            continue
    st.error("AI request failed after retries.")
    return "Unable to get AI response."

@st.cache_data
def train_data(stock_symbols, start='2000-01-01', end=str(date.today())):
    """Download and cache stock data from yfinance."""
    try:
        if isinstance(stock_symbols, str):
            stock_symbols = [stock_symbols]
        data = yf.download(stock_symbols, start=start, end=end, progress=False)
        if data.empty:
            raise ValueError("No data retrieved.")
        if len(stock_symbols) == 1:
            data = data.reset_index()
        else:
            data = data['Close'].reset_index()
        return data
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        return pd.DataFrame()

def save_to_csv(stock_name, operation, df):
    """Save DataFrame to CSV with error handling."""
    os.makedirs('csvfile', exist_ok=True)
    file_path = f'csvfile/{stock_name}_NS_{operation}.csv'
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        st.error(f"Failed to save CSV {file_path}: {e}")

def create_data(operation, **kwargs):
    """Create DataFrame for stock data based on operation."""
    base_data = {'Date': str(date.today())}
    if operation == 'close':
        base_data.update({'Price': 0, 'High': kwargs['high'], 'Low': kwargs['low'],
                         'Open': kwargs['open'], 'Volume': kwargs['volume']})
    elif operation == 'high':
        base_data.update({'Price': 0, 'Close': kwargs['close'], 'Low': kwargs['low'],
                         'Open': kwargs['open'], 'Volume': kwargs['volume']})
    elif operation == 'low':
        base_data.update({'Price': 0, 'Close': kwargs['close'], 'High': kwargs['high'],
                         'Open': kwargs['open'], 'Volume': kwargs['volume']})
    return pd.DataFrame([base_data])

def handle_operation(stock_name, stock_symbol, operation):
    """Handle stock data operations (close, high, low) with stable inputs."""
    file_path = f'csvfile/{stock_name}_NS_{operation}.csv'
    
    # Check if file already exists
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if df['Date'].iloc[-1] == str(date.today()):
            predict_price(df, operation, stock_name, stock_symbol)
        else:
            os.remove(file_path)
            st.warning("⚠️ Existing data is outdated. Please enter new stock data.")
            handle_operation(stock_name, stock_symbol, operation)
    else:
        st.info("📝 No data found. Please enter stock values for prediction.")
        
        # Collect user inputs using the fixed input function
        inputs = collect_user_inputs(operation)
        
        # Now inputs will return only after button is pressed and values are filled
        if inputs:
            df = create_data(operation, **inputs)
            save_to_csv(stock_name, operation, df)
            st.success("✅ Data saved. Running prediction...")
            predict_price(df, operation, stock_name, stock_symbol)
def collect_user_inputs(operation):
    """Step-by-step input collection for stock data based on operation (close, high, low)."""
    
    prefix = f"{operation}_step_"
    
    # Initialize step tracker
    if f"{prefix}step" not in st.session_state:
        st.session_state[f"{prefix}step"] = 1

    step = st.session_state[f"{prefix}step"]

    # Define the input sequence based on operation
    input_order = {
        'close': ['High', 'Low', 'Open', 'Volume'],
        'high': ['Close', 'Low', 'Open', 'Volume'],
        'low': ['Close', 'High', 'Open', 'Volume']
    }

    labels = {
        'High': "High value",
        'Low': "Low value",
        'Open': "Open value",
        'Close': "Close value",
        'Volume': "Volume"
    }

    fields = input_order.get(operation, [])

    # Show one input field at a time
    if step <= len(fields):
        field_name = fields[step - 1]
        key = f"{prefix}{field_name}"
        value = st.number_input(
            f"Step {step}️⃣: Enter {labels[field_name]}",
            min_value=0.0,
            key=key
        )
        if st.button("Next ➡️", key=f"{key}_next") and value > 0:
            st.session_state[f"{prefix}step"] += 1
            st.experimental_rerun()

    # Once all fields are entered, show submit button
    if step > len(fields):
        if st.button("✅ Submit All", key=f"{prefix}submit"):
            inputs = {}
            for f in fields:
                inputs[f] = st.session_state.get(f"{prefix}{f}", 0.0)
            if all(v > 0 for v in inputs.values()):
                 return inputs
            else:
                st.warning("⚠️ Please enter all values greater than 0.")

    return None

def predict_price(df, operation, stock_name, stock_symbol):
    """Predict stock price using LinearRegression."""
    stock_data = train_data(stock_symbol)
    if stock_data.empty:
        return
    df = df.drop(columns=['Date'], errors='ignore')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    target_col = operation.capitalize()
    X = stock_data.drop(columns=[target_col, 'Date'], errors='ignore')
    y = stock_data[target_col]
    lr = LinearRegression()
    lr.fit(X, y)
    actual_price = stock_data[target_col].tail(1).values[0]
    predicted_price = lr.predict(df.drop(columns=['Price'], errors='ignore'))[0]
    ai_prompt = (f"Hello AI, actual price: {actual_price:.2f}, predicted price: {predicted_price:.2f}. "
                 f"Summarize the stock {stock_name} and advise on buy/sell.")
    st.success(call_ai(ai_prompt))

def predict_time_range(stock_name, stock_symbol, operation, time_range):
    """Handle predictions for different time ranges."""
    periods = {'week': 1040, 'month': 240, 'year': 20}
    if time_range not in periods:
        st.error("Invalid time range.")
        return
    data = train_data(stock_symbol)
    if data.empty:
        return
    chunk_size = int(len(data) / periods[time_range])
    aggregated_data = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size]
        agg = {
            'close': np.average(chunk['Close']),
            'high': np.average(chunk['High']),
            'low': np.average(chunk['Low']),
            'volume': np.average(chunk['Volume'])
        }
        aggregated_data.append(agg)
    df = pd.DataFrame(aggregated_data)
    X = df.drop(columns=[operation], errors='ignore')
    y = df[operation]
    lr = LinearRegression()
    lr.fit(X, y)
    tail_size = chunk_size
    pred_input = {
        'close': np.average(data['Close'].tail(tail_size)),
        'high': np.average(data['High'].tail(tail_size)),
        'low': np.average(data['Low'].tail(tail_size)),
        'volume': np.average(data['Volume'].tail(tail_size))
    }
    pred_df = pd.DataFrame([pred_input]).drop(columns=[operation], errors='ignore')
    predicted_price = lr.predict(pred_df)[0]
    actual_price = data[operation.capitalize()].tail(1).values[0]
    csv_file = f'csvfile/data_{time_range}.csv'
    new_row = {
        'index': pd.read_csv(csv_file)['index'].max() + 1 if os.path.exists(csv_file) else 1,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S'),
        'stock_name': stock_name,
        'operation': operation,
        'predicted_value': predicted_price,
        'actual_value': actual_price
    }
    df_new = pd.DataFrame([new_row])
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_new = pd.concat([df_existing, df_new], ignore_index=True)
    df_new.to_csv(csv_file, index=False)
    ai_prompt = (f"Hello AI, predicted price: {predicted_price:.2f}. "
                 f"Summarize the stock {stock_name} and advise on buy/sell.")
    st.success(call_ai(ai_prompt))

def optimize_portfolio(tickers, total_investment, investment_duration, start_date, end_date):
    """Optimize portfolio using PyPortfolioOpt."""
    try:
        # Fetch data
        data = train_data(tickers, start=start_date, end=end_date)
        if data.empty:
            st.error("No data available for selected stocks.")
            return
        
        # Calculate expected returns and covariance
        mu = expected_returns.mean_historical_return(data.set_index('Date'))
        S = risk_models.sample_cov(data.set_index('Date'))
        
        # Optimize portfolio
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        # Display allocation
        st.subheader("Optimized Portfolio Allocation")
        allocation_data = [
            {
                'Stock': stock,
                'Weight (%)': round(weight * 100, 2),
                'Amount (₹)': round(weight * total_investment, 2)
            }
            for stock, weight in cleaned_weights.items()
        ]
        st.dataframe(pd.DataFrame(allocation_data))
        
        # Portfolio performance
        st.subheader("Portfolio Performance")
        performance = ef.portfolio_performance(verbose=False)
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Annual Return", f"{performance[0] * 100:.2f}%")
        col2.metric("Annual Volatility", f"{performance[1] * 100:.2f}%")
        col3.metric("Sharpe Ratio", f"{performance[2]:.2f}")
        
        # Time period info
        years = pd.to_datetime(end_date).year - pd.to_datetime(start_date).year
        st.write(f"Data used from {start_date} to {end_date} ({years} years)")
        
        # Pie chart
        st.subheader("Portfolio Allocation Chart")
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.pie(
            list(cleaned_weights.values()),
            labels=list(cleaned_weights.keys()),
            autopct='%1.1f%%',
            startangle=140
        )
        ax.set_title('Optimized Portfolio Allocation')
        ax.axis('equal')
        st.pyplot(fig)
        
        # Estimated profit
        st.subheader("Estimated Profit")
        future_value = total_investment * (1 + performance[0]) ** investment_duration
        profit = future_value - total_investment
        col1, col2 = st.columns(2)
        col1.metric("Future Value (₹)", f"{future_value:.2f}")
        col2.metric("Profit (₹)", f"{profit:.2f}")
        
        # AI summary
        ai_prompt = (f"Hello AI, I have a portfolio with {', '.join(tickers)} optimized for maximum Sharpe ratio. "
                     f"Expected return: {performance[0] * 100:.2f}%, volatility: {performance[1] * 100:.2f}%. "
                     f"Summarize the portfolio and advise on investment.")
        st.success(call_ai(ai_prompt))
        
    except Exception as e:
        st.error(f"Portfolio optimization failed: {e}")

def question_recognizer():
    """Handle AI prompt input from sidebar."""
    st.sidebar.header("AI Assistant")
    query = st.sidebar.text_input("Enter your prompt:", key="ai_query")
    if st.sidebar.button("Submit Query", key="submit_query"):
        if query.strip():
            st.success(call_ai(query))
        else:
            st.success(call_ai())

def portfolio_section():
    """Portfolio optimization section."""
    st.header("Portfolio Optimization")
    st.write("Optimize a portfolio of stocks for maximum Sharpe ratio.")
    
    # User inputs
    selected_stocks = st.multiselect(
        "Select stocks for portfolio:",
        options=list(stock_name_dict.keys()),
        key="portfolio_stocks"
    )
    total_investment = st.number_input("Total investment amount (₹):", min_value=0.0, value=1000.0)
    investment_duration = st.number_input("Investment duration (years):", min_value=1, value=3)
    start_date = st.date_input("Start date:", value=pd.to_datetime("2001-01-01"))
    end_date = st.date_input("End date:", value=pd.to_datetime("2025-01-01"))
    
    if st.button("Optimize Portfolio", key="optimize_portfolio"):
        if not selected_stocks:
            st.error("Please select at least one stock.")
            return
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            return
        # Map stock names to symbols
        tickers = [stock_name_dict.get(stock.lower()) for stock in selected_stocks]
        if None in tickers:
            st.error("One or more selected stocks are invalid.")
            return
        optimize_portfolio(tickers, total_investment, investment_duration, str(start_date), str(end_date))

def stock_prediction_section():
    """Stock prediction section."""
    st.header("Stock Price Prediction")
    selected_stock = st.sidebar.selectbox("Select stock", ['Select'] + list(stock_name_dict.keys()), key="stock_select")
    time_range = st.sidebar.selectbox("Select time range", ['Select', 'day', 'week', 'month', 'year'], key="time_range")
    operation = st.sidebar.selectbox("Select data type", ['Select', 'close', 'high', 'low'], key="operation")

    if st.sidebar.button("Analyze", key="analyze_stock"):
        if selected_stock == 'Select' or time_range == 'Select' or operation == 'Select':
            st.error("Please select a stock, time range, and operation.")
            return
        stock_symbol = stock_name_dict.get(selected_stock.lower())
        if not stock_symbol:
            st.error("Invalid stock selected.")
            return
        if time_range == 'day':
            handle_operation(selected_stock, stock_symbol, operation)
        else:
            predict_time_range(selected_stock, stock_symbol, operation, time_range)

def transcribe_audio(path):
    """Transcribe audio file using faster_whisper."""
    if not whisper_model:
        st.error("Whisper model unavailable.")
        return ""
    try:
        segments, info = whisper_model.transcribe(path, beam_size=5)
        lang = info.language
        full_text = " ".join([seg.text for seg in segments])
        return full_text,lang
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return ","

def news_summarizer_section():
    """News summarizer section for YouTube video summaries."""
    st.header("YouTube News Summarizer")
    st.write("Enter a YouTube video URL to summarize stock-related news.")
    
    video_url = st.text_input("Enter YouTube Video URL:", key="news_video_url")
    
    if st.button("Process Video", key="process_video"):
        if not video_url.strip():
            st.error("Please enter a valid YouTube URL.")
            return
        
        audio_path = "audio.mp3"
        
        # Step 1: Download audio using yt-dlp
        st.write("📥 Downloading audio...")
        try:
            command = ["yt-dlp", "-x", "--audio-format", "mp3", "-o", audio_path, video_url]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                st.error(f"Failed to download audio: {result.stderr}")
                return
        except Exception as e:
            st.error(f"Audio download failed: {e}")
            return
        
        # Step 2: Transcribe audio
        st.write("🎙 Transcribing audio...")
        transcript, detected_lang = transcribe_audio(audio_path)
        st.info(f"🈶 Detected Language: {detected_lang.upper()}")
        transcript = translate_to_english(transcript, detected_lang)

        if not transcript:
            st.error("Transcription failed. Please try another video.")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return
        st.write("Transcript (preview):", transcript[:1000], "...")
        
        # Step 3: Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Step 4: Generate summary using existing AI function
        st.write("📝 Generating Summary...")
        summary_prompt = f"Summarize the following news content related to stocks or markets:\n\n{transcript}"
        summary = call_ai(summary_prompt)
        if summary == "Unable to get AI response.":
            st.error("Failed to generate summary.")
        else:
            st.success("✅ Summary Ready!")
            st.write(summary)

            # Step 5: Sentiment scoring
            st.write("🧠 Analyzing Sentiment...")
            cleaned_summary = clean_text(summary)
            score = get_sentiment_score(cleaned_summary)

            if score == 1:
                st.success("📊 Sentiment: Positive")
            elif score == 0:
                st.warning("📊 Sentiment: Neutral")
            else:
                st.error("📊 Sentiment: Negative")

def get_sentiment_score(text):
    """Get sentiment score from cleaned news summary."""
    result = sentiment_model(text[:512])[0]  # Limit to 512 chars
    label = result['label']
    if "1" in label or "2" in label:
        return -1  # Negative
    elif "3" in label:
        return 0   # Neutral
    else:
        return 1   # Positive

def clean_text(text):
    """Clean up raw transcript for sentiment scoring."""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def translate_to_english(text, lang_code):
    """Use Gemini API to translate any language text to English."""
    if lang_code == "en":
        return text
    prompt = f"Translate this text from {lang_code} to English:\n\n{text}"
    return call_ai(prompt)

def main():
    """Main application logic."""
    st.title("MARKET.AI")
    st.sidebar.header("Navigation")
    
    section = st.sidebar.radio("Select section", ["Stock Prediction", "Portfolio Optimization", "News Summarizer", "AI Assistant"])
    
    if section == "Stock Prediction":
        stock_prediction_section()
    elif section == "Portfolio Optimization":
        portfolio_section()
    elif section == "News Summarizer":
        news_summarizer_section()
    else:
        question_recognizer()

if __name__ == "__main__":
    main()
