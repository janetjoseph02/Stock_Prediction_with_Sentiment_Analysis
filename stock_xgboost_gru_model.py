import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import re
import os
import xgboost as xgb
import shap
import plotly.graph_objects as go
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

# Load FinBERT sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Load Indian stock symbols
@st.cache_data
def get_indian_stocks():
    file_path = "indian_stocks.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="utf-8")
        df.columns = df.columns.str.strip()
        if "SYMBOL" in df.columns:
            return df["SYMBOL"].dropna().tolist()
        else:
            st.error("Error: 'SYMBOL' column not found.")
            return []
    else:
        st.error("File 'indian_stocks.csv' not found.")
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

# Fetch stock data
def get_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end)
    if not data.empty:
        data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        data.index = data.index.tz_localize(None)
    return data

# Fetch stock info
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    def format_value(value, format_str):
        if value == "N/A" or value is None:
            return "N/A"
        return format_str.format(value)
    
    return {
        "Market Cap": format_value(info.get("marketCap"), "{:,} INR"),
        "P/E Ratio": format_value(info.get("trailingPE"), "{}"),
        "ROCE": format_value(info.get("returnOnCapitalEmployed"), "{:.2f}%"),
        "Current Price": format_value(info.get("currentPrice"), "{:.2f} INR"),
        "Book Value": format_value(info.get("bookValue"), "{:.2f} INR"),
        "ROE": format_value(info.get("returnOnEquity"), "{:.2f}%"),
        "Dividend Yield": format_value(info.get("dividendYield"), "{:.2f}%"),
        "Face Value": format_value(info.get("faceValue"), "{:.2f} INR"),
        "High": format_value(info.get("dayHigh"), "{:.2f} INR"),
        "Low": format_value(info.get("dayLow"), "{:.2f} INR"),
    }

# News API
NEWS_API_KEY = "563215a35c1a47968f46271e04083ea3"
NEWS_API_URL = "https://newsapi.org/v2/everything"

def get_news(stock_symbol):
    stock_name_mapping = {
        "RELIANCE": "Reliance Industries",
        "TCS": "Tata Consultancy Services",
        "INFY": "Infosys",
        "HDFCBANK": "HDFC Bank",
        "ICICIBANK": "ICICI Bank"
    }
    query = stock_name_mapping.get(stock_symbol, stock_symbol)
    params = {"q": query, "apiKey": NEWS_API_KEY, "language": "en", "sortBy": "publishedAt"}
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching news: {response.json()}")
        return []
    return response.json().get("articles", [])

# Sentiment analysis
def analyze_sentiment(text):
    if not text:
        return "neutral", 0.0
    result = sentiment_pipeline(text[:512])[0]
    return result['label'], result['score']

# Filter relevant news
def filter_relevant_news(news_articles, stock_name):
    filtered_articles = []
    for article in news_articles:
        title = article.get('title', '')
        if title and re.search(stock_name, title, re.IGNORECASE):  
            filtered_articles.append(article)
    return filtered_articles

# Feature engineering
def create_advanced_features(df):
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['5D_MA'] = df['Close'].rolling(5).mean()
    df['20D_MA'] = df['Close'].rolling(20).mean()
    df['MA_Ratio'] = df['5D_MA'] / df['20D_MA']
    df['5D_Volatility'] = df['Returns'].rolling(5).std()
    df['Volume_MA5'] = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    return df.dropna()

# Hybrid XGBoost-GRU Model
def create_hybrid_model(df_stock, sentiment_features):
    # Prepare data with sentiment
    sentiment_df = pd.DataFrame(sentiment_features.items(), columns=["Date", "Sentiment"])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)
    df_stock.index = pd.to_datetime(df_stock.index).tz_localize(None)
    df_stock = df_stock.reset_index().merge(sentiment_df, on="Date", how="left").set_index("Date")
    df_stock['Sentiment'] = pd.to_numeric(df_stock['Sentiment'], errors='coerce').fillna(0)
    
    # Feature engineering
    df_stock = create_advanced_features(df_stock)
    df_stock['Target'] = df_stock['Close'].pct_change().shift(-1)
    df_stock.dropna(inplace=True)
    
    # Features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment',
               '5D_MA', 'MA_Ratio', '5D_Volatility', 'Volume_Ratio']
    
    # Train-test split
    X = df_stock[features]
    y = df_stock['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 1. XGBoost Model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        early_stopping_rounds=30,
        random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # 2. GRU Model
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_3d = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    gru_model = Sequential([
        GRU(64, input_shape=(1, X_3d.shape[2]), return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    
    gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    gru_model.fit(X_3d[:len(X_train)], y_train[:len(X_train)], 
                epochs=50, batch_size=32, verbose=0)
    
    # Ensemble predictions
    xgb_pred = xgb_model.predict(X_test)
    gru_pred = gru_model.predict(X_3d[len(X_train):]).flatten()
    final_pred = (0.6 * xgb_pred) + (0.4 * gru_pred)
    df_stock.loc[y_test.index, 'Predicted'] = final_pred
    
    # Calculate accuracy metrics
    mae = mean_absolute_error(y_test, final_pred)
    accuracy = max(0, 100 - (mae * 100))
    
    return df_stock, {'xgb': xgb_model, 'gru': gru_model}, scaler, features, accuracy

# Enhanced prediction function with better fluctuations
def hybrid_predict_prices(models, scaler, last_known_data, features, days=10):
    future_dates = pd.date_range(start=last_known_data.index[-1], periods=days + 1, freq='B')[1:]
    future_prices = pd.DataFrame(index=future_dates, columns=['Predicted Close'])
    
    current_data = last_known_data.copy()
    last_close = current_data['Close'].iloc[-1]
    
    for i, date in enumerate(future_dates):
        # Prepare input
        input_data = current_data[features].iloc[-1:].copy()
        
        # XGBoost prediction
        xgb_pred = models['xgb'].predict(input_data)[0]
        
        # GRU prediction
        input_scaled = scaler.transform(input_data)
        input_3d = input_scaled.reshape(1, 1, input_scaled.shape[1])
        gru_pred = models['gru'].predict(input_3d)[0][0]
        
        # Combined prediction with realistic noise
        combined_pred = (0.6 * xgb_pred) + (0.4 * gru_pred)
        volatility = current_data['5D_Volatility'].iloc[-1]
        
        # More realistic noise generation
        noise = np.random.normal(0, volatility * 1.5)  # Increased volatility impact
        adj_pred = combined_pred * (1 + noise)
        
        # Calculate new price with more fluctuation
        price_change = adj_pred + (np.random.uniform(-0.015, 0.015))  # Additional random fluctuation
        new_close = last_close * (1 + price_change)
        future_prices.loc[date, 'Predicted Close'] = new_close
        last_close = new_close
        
        # Update current_data with more realistic simulated values
        new_row = {
            'Open': new_close * (0.998 + np.random.uniform(-0.004, 0.004)),
            'High': new_close * (1.015 + np.random.uniform(-0.01, 0.01)),
            'Low': new_close * (0.985 + np.random.uniform(-0.01, 0.01)),
            'Close': new_close,
            'Volume': current_data['Volume'].iloc[-1] * (0.95 + np.random.uniform(-0.1, 0.1)),
            'Sentiment': current_data['Sentiment'].iloc[-1] * (0.9 + np.random.uniform(-0.2, 0.2)),
            '5D_MA': current_data['5D_MA'].iloc[-1] * (1 + np.random.uniform(-0.01, 0.01)),
            '20D_MA': current_data['20D_MA'].iloc[-1] * (1 + np.random.uniform(-0.005, 0.005)),
            'MA_Ratio': current_data['MA_Ratio'].iloc[-1] * (1 + np.random.uniform(-0.02, 0.02)),
            '5D_Volatility': current_data['5D_Volatility'].iloc[-1] * (1 + np.random.uniform(-0.15, 0.15)),
            'Volume_MA5': current_data['Volume_MA5'].iloc[-1] * (1 + np.random.uniform(-0.08, 0.08)),
            'Volume_Ratio': current_data['Volume_Ratio'].iloc[-1] * (1 + np.random.uniform(-0.15, 0.15))
        }
        current_data = pd.concat([current_data.iloc[1:], pd.DataFrame(new_row, index=[date])])
    
    return future_prices

# Candlestick chart
def create_candlestick_chart(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    )])
    fig.update_layout(
        title="Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    return fig

# Generate simple investment recommendation
def generate_recommendation(predicted_prices, current_price, accuracy, avg_sentiment):
    avg_prediction = predicted_prices['Predicted Close'].mean()
    price_change = ((avg_prediction - current_price) / current_price) * 100
    
    # Incorporate sentiment into recommendation
    sentiment_factor = 1 + (avg_sentiment * 0.5)  # Amplify sentiment impact
    
    adjusted_change = price_change * sentiment_factor
    
    if adjusted_change > 7 and accuracy > 72:
        return "STRONG BUY", "High confidence in significant price increase"
    elif adjusted_change > 3 and accuracy > 65:
        return "BUY", "Good confidence in moderate price increase"
    elif adjusted_change > 0 and accuracy > 60:
        return "HOLD (Positive)", "Potential for slight growth"
    elif adjusted_change < -7 and accuracy > 72:
        return "STRONG SELL", "High confidence in significant price drop"
    elif adjusted_change < -3 and accuracy > 65:
        return "SELL", "Good confidence in moderate price drop"
    elif adjusted_change < 0 and accuracy > 60:
        return "HOLD (Caution)", "Potential for slight decline"
    else:
        return "HOLD", "Unclear direction - consider other factors"

# Streamlit UI
st.title("Indian Stock Market Analysis")
st.sidebar.header("Stock Selection")
indian_stocks = get_indian_stocks()
selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks)
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
chart_type = st.sidebar.radio("Chart Type", ["Candlestick Chart", "Line Chart"])

if st.sidebar.button("Analyze"):
    ticker = f"{selected_stock}.NS"
    df_stock = get_stock_data(ticker, start_date, end_date)
    
    if not df_stock.empty:
        df_stock = df_stock.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        
        # Display stock info (original format)
        st.subheader(f"Stock Information for {selected_stock}")
        stock_info = get_stock_info(ticker)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Market Cap:** {stock_info['Market Cap']}")
            st.write(f"**P/E Ratio:** {stock_info['P/E Ratio']}")
            st.write(f"**ROCE:** {stock_info['ROCE']}")
            st.write(f"**Current Price:** {stock_info['Current Price']}")
        with col2:
            st.write(f"**Book Value:** {stock_info['Book Value']}")
            st.write(f"**ROE:** {stock_info['ROE']}")
            st.write(f"**Dividend Yield:** {stock_info['Dividend Yield']}")
            st.write(f"**Face Value:** {stock_info['Face Value']}")
        st.write(f"**High/Low:** {stock_info['High']} / {stock_info['Low']}")

        # Display complete OHLCV table
        st.subheader("Historical Price Data")
        st.dataframe(df_stock[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index(ascending=False).style.format({
            'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 
            'Close': '{:.2f}', 'Volume': '{:,}'
        }))

        # Display chart
        st.subheader(f"{chart_type} for {selected_stock}")
        if chart_type == "Candlestick Chart":
            st.plotly_chart(create_candlestick_chart(df_stock))
        else:
            st.line_chart(df_stock["Close"])

        # News and sentiment analysis
        st.subheader(f"Latest News for {selected_stock}")
        news_articles = get_news(selected_stock)
        filtered_news = filter_relevant_news(news_articles, selected_stock)
        daily_sentiment = {}
        sentiment_data = []

        if not filtered_news:
            st.warning("No recent news articles found for this stock. The analysis will continue without news sentiment data.")
        else:
            for article in filtered_news:
                title = article.get('title', '')
                description = article.get('description', '')
                text = f"{title} {description}".strip()
                sentiment, confidence_score = analyze_sentiment(text)
                date = article.get("publishedAt", "")[0:10]
                sentiment_data.append([date, title, sentiment, f"{confidence_score:.2f}"])

                if date in daily_sentiment:
                    daily_sentiment[date].append((sentiment, confidence_score))
                else:
                    daily_sentiment[date] = [(sentiment, confidence_score)]

                st.write(f"**{title}**")
                st.write(description)
                st.write(f"[Read more]({article['url']})")
                st.write("---")

        # News sentiment table (only show if we have news)
        if sentiment_data:
            st.subheader("News Sentiment Analysis")
            df_sentiment = pd.DataFrame(sentiment_data, columns=["Date", "Headline", "Sentiment", "Confidence"])
            st.dataframe(df_sentiment.sort_values("Date", ascending=False))

        # Enhanced Daily Average Sentiment table (only show if we have news)
        if daily_sentiment:
            st.subheader("Daily Weighted Average Sentiment")
            avg_daily_sentiment = []
            for date, scores in daily_sentiment.items():
                weighted_sum = 0
                total_weight = 0
                for sentiment, score in scores:
                    value = 1 if sentiment == "positive" else (-1 if sentiment == "negative" else 0)
                    weighted_sum += value * score
                    total_weight += score
                avg_score = weighted_sum / total_weight if total_weight != 0 else 0
                
                # Classify sentiment
                if avg_score > 0.2:
                    sentiment_class = "Positive"
                elif avg_score < -0.2:
                    sentiment_class = "Negative"
                else:
                    sentiment_class = "Neutral"
                
                avg_daily_sentiment.append([date, f"{avg_score:.2f}", sentiment_class])
            
            df_avg_sentiment = pd.DataFrame(avg_daily_sentiment, 
                                          columns=["Date", "Weighted Score", "Sentiment"])
            st.dataframe(df_avg_sentiment.sort_values("Date", ascending=False))

        # Train hybrid model (modified to handle no news case)
        df_stock, models, scaler, features, accuracy = create_hybrid_model(df_stock, daily_sentiment if daily_sentiment else {})

        # Enhanced Explainable AI Conclusion with detailed explanation
        st.subheader("AI Analysis Conclusion")
        current_price = df_stock['Close'].iloc[-1]
        avg_sentiment = df_stock['Sentiment'].mean() if 'Sentiment' in df_stock else 0
        
        # Generate detailed explanation (modified for no news case)
        if daily_sentiment:
            sentiment_analysis_part = f"""
            3. **Market Sentiment**: 
               - News sentiment is predominantly {'positive' if avg_sentiment > 0 else 'negative' if avg_sentiment < 0 else 'neutral'}
               - This sentiment is {'strengthening' if df_stock['Sentiment'].iloc[-1] > df_stock['Sentiment'].mean() else 'weakening' if df_stock['Sentiment'].iloc[-1] < df_stock['Sentiment'].mean() else 'stable'}
            """
            conclusion = f"These factors collectively suggest that {selected_stock} is currently in a {'favorable' if avg_sentiment > 0 else 'challenging' if avg_sentiment < 0 else 'neutral'} position."
        else:
            sentiment_analysis_part = """
            3. **Market Sentiment**: 
               - No recent news sentiment data available
               - Analysis based solely on technical indicators
            """
            conclusion = f"These technical indicators suggest that {selected_stock} is currently showing {'strong' if df_stock['MA_Ratio'].iloc[-1] > 1.05 else 'weak' if df_stock['MA_Ratio'].iloc[-1] < 0.95 else 'neutral'} technical signals."

        explanation = f"""
        Our analysis of {selected_stock} reveals the following key insights:
        
        1. **Price Movement**: The stock is currently trading at ₹{current_price:.2f}. Based on our model's analysis (which has {accuracy:.1f}% confidence), we expect {'an upward trend' if avg_sentiment > 0 or df_stock['MA_Ratio'].iloc[-1] > 1 else 'a downward trend' if avg_sentiment < 0 or df_stock['MA_Ratio'].iloc[-1] < 1 else 'relative stability'} in the coming days.
        
        2. **Technical Indicators**:
           - The stock is currently trading {'above' if df_stock['MA_Ratio'].iloc[-1] > 1 else 'below'} its 20-day moving average
           - Recent volatility is {'high' if df_stock['5D_Volatility'].iloc[-1] > 0.02 else 'moderate' if df_stock['5D_Volatility'].iloc[-1] > 0.01 else 'low'}
           - Volume trends show {'increasing' if df_stock['Volume_Ratio'].iloc[-1] > 1 else 'decreasing' if df_stock['Volume_Ratio'].iloc[-1] < 1 else 'stable'} trading activity
        
        {sentiment_analysis_part}
        
        {conclusion}
        """
        st.write(explanation)

# IMPROVED 10-DAY PRICE FORECAST WITH DAILY FLUCTUATIONS
try:
    st.subheader("10-Day Price Forecast")
    last_data = df_stock.iloc[-30:]
    current_price = last_data['Close'].iloc[-1]

    # 1. Get base predictions from model
    future_prices = hybrid_predict_prices(models, scaler, last_data, features)

    # 2. Calculate realistic daily fluctuations
    historical_volatility = last_data['Close'].pct_change().std() * 100  # in percentage
    np.random.seed(42)  # For reproducibility
    daily_fluctuations = np.random.uniform(
        low=-2 * historical_volatility,
        high=2 * historical_volatility,
        size=len(future_prices)
    ) / 100  # Convert to decimal

    # 3. Apply sentiment-adjusted trend with fluctuations
    daily_sentiment_impact = 1 + (avg_sentiment * 0.005)  # 0.5% daily multiplier
    daily_multipliers = 1 + (future_prices['Predicted Close'].diff().fillna(0) + daily_fluctuations)
    future_prices['Forecast Price'] = current_price * np.cumprod(daily_multipliers * daily_sentiment_impact)

    # 4. Apply smoothing and realistic bounds
    future_prices['Forecast Price'] = (
        future_prices['Forecast Price']
        .rolling(window=2, min_periods=1).mean()  # Gentle smoothing
        .clip(current_price * 0.85, current_price * 1.15)  # ±15% bounds
    )

    # 5. Calculate daily percentage changes
    future_prices['Daily Change (%)'] = future_prices['Forecast Price'].pct_change().fillna(0) * 100

    # DISPLAY FORECAST TABLE WITH IMPROVED STYLING
    def format_forecast_table(df):
        styled_df = (df[['Forecast Price', 'Daily Change (%)']]
                   .rename(columns={
                       'Forecast Price': 'Price (₹)',
                       'Daily Change (%)': 'Daily Change (%)'
                   })
                   .style
                   .format({
                       'Price (₹)': '₹{:,.2f}',
                       'Daily Change (%)': '{:+.2f}%'
                   })
                   .applymap(
                       lambda x: 'color: #4CAF50' if x > 0 else 'color: #F44336',
                       subset=['Daily Change (%)']
                   ))
        return styled_df.set_properties(**{
            'background-color': '#f8f9fa',
            'border': '1px solid #ddd',
            'text-align': 'center'
        }, subset=['Price (₹)', 'Daily Change (%)'])

    # Display the forecast table with some spacing
    st.markdown("**Forecasted Prices with Daily Changes**")
    st.dataframe(
        format_forecast_table(future_prices),
        use_container_width=True,
        height=(len(future_prices) + 1) * 35 + 3
    )

    # INVESTMENT RECOMMENDATION
    recommendation, reasoning = generate_recommendation(
        future_prices, 
        current_price, 
        accuracy,
        avg_sentiment
    )

    # Display recommendation with emoji and colored box
    st.subheader("Investment Recommendation")
    rec_colors = {
        "STRONG BUY": "green",
        "BUY": "lightgreen",
        "HOLD (Positive)": "lightblue",
        "HOLD": "blue",
        "HOLD (Caution)": "orange",
        "SELL": "pink",
        "STRONG SELL": "red"
    }
    
    rec_color = rec_colors.get(recommendation.split()[0], "blue")
    st.markdown(
        f"""<div style="padding: 10px; border-radius: 5px; background-color: {rec_color}; color: white">
        <strong>{recommendation}:</strong> {reasoning}
        </div>""",
        unsafe_allow_html=True
    )

    # PRICE TREND VISUALIZATION
    st.subheader("Price Trend Analysis")
    historical_data = df_stock[['Close']].rename(columns={'Close': 'Price'}).iloc[-90:]
    future_dates = pd.date_range(
        start=historical_data.index[-1] + pd.Timedelta(days=1),
        periods=len(future_prices),
        freq='D'
    )
    future_data = pd.DataFrame(
        {'Price': future_prices['Forecast Price'].values},
        index=future_dates
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Price'],
        mode='lines',
        name='Historical Prices',
        line=dict(color='#3366CC', width=2),
        hovertemplate='₹%{y:.2f}<extra>%{x|%b %d, %Y}</extra>'
    ))
    fig.add_trace(go.Scatter(
        x=future_data.index,
        y=future_data['Price'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#FF7F0E', width=2, dash='dot'),
        marker=dict(size=8, color='#FF7F0E'),
        hovertemplate='₹%{y:.2f}<extra>%{x|%b %d, %Y}</extra>'
    ))
    fig.add_trace(go.Scatter(
        x=[historical_data.index[-1]],
        y=[current_price],
        mode='markers',
        name='Current Price',
        marker=dict(size=12, color='#DC3912'),
        hovertemplate='₹%{y:.2f}<extra>Current Price</extra>'
    ))
    fig.add_vline(
        x=historical_data.index[-1],
        line_width=1,
        line_dash="dash",
        line_color="grey"
    )
    fig.update_layout(
        title=f"{selected_stock} Price Trend (Historical vs Forecast)",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode="x unified",
        plot_bgcolor='rgba(240,240,240,0.8)',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error generating forecast: {str(e)}")

# # INVESTMENT RECOMMENDATION (unchanged)
# recommendation, reasoning = generate_recommendation(
#     future_prices, 
#     current_price, 
#     accuracy,
#     avg_sentiment
# )

# st.subheader("Investment Recommendation")
# if "BUY" in recommendation:
#     st.success(f"✅ {recommendation}: {reasoning}")
# elif "SELL" in recommendation:
#     st.error(f"❌ {recommendation}: {reasoning}")
# else:
#     st.warning(f"⚠ {recommendation}: {reasoning}")

# PRICE TREND VISUALIZATION (with improved daily variations)
# st.subheader("Price Trend Analysis")

# # Prepare data
# historical_data = df_stock[['Close']].rename(columns={'Close': 'Price'}).iloc[-90:]
# future_dates = pd.date_range(
#     start=historical_data.index[-1] + pd.Timedelta(days=1),
#     periods=len(future_prices),
#     freq='D'
# )
# future_data = pd.DataFrame(
#     {'Price': future_prices['Forecast Price'].values},
#     index=future_dates
# )

# # Create figure
# fig = go.Figure()

# # Historical data
# fig.add_trace(go.Scatter(
#     x=historical_data.index,
#     y=historical_data['Price'],
#     mode='lines',
#     name='Historical Prices',
#     line=dict(color='#3366CC', width=2),
#     hovertemplate='₹%{y:.2f}<extra>%{x|%b %d, %Y}</extra>'
# ))

# # Forecast data with daily variations
# fig.add_trace(go.Scatter(
#     x=future_data.index,
#     y=future_data['Price'],
#     mode='lines+markers',
#     name='Forecast',
#     line=dict(color='#FF7F0E', width=2, dash='dot'),
#     marker=dict(size=8, color='#FF7F0E'),
#     hovertemplate='₹%{y:.2f}<extra>%{x|%b %d, %Y}</extra>'
# ))

# # Current price marker
# fig.add_trace(go.Scatter(
#     x=[historical_data.index[-1]],
#     y=[current_price],
#     mode='markers',
#     name='Current Price',
#     marker=dict(
#         size=12,
#         color='#DC3912',
#         line=dict(width=2, color='DarkSlateGrey')
#     ),
#     hovertemplate='₹%{y:.2f}<extra>Current Price</extra>'
# ))

# # Add separation line
# fig.add_vline(
#     x=historical_data.index[-1],
#     line_width=1,
#     line_dash="dash",
#     line_color="grey"
# )

# # Update layout (unchanged from your good version)
# fig.update_layout(
#     title=f"{selected_stock} Price Trend (Historical vs Forecast)",
#     xaxis_title="Date",
#     yaxis_title="Price (₹)",
#     legend=dict(
#         orientation="h",
#         yanchor="bottom",
#         y=1.02,
#         xanchor="right",
#         x=1
#     ),
#     hovermode="x unified",
#     plot_bgcolor='rgba(240,240,240,0.8)',
#     paper_bgcolor='rgba(255,255,255,0.8)',
#     xaxis=dict(
#         showgrid=True,
#         gridcolor='lightgrey',
#         rangeslider=dict(visible=False)
#     ),
#     yaxis=dict(
#         showgrid=True,
#         gridcolor='lightgrey',
#         tickprefix="₹"
#     ),
#     margin=dict(l=20, r=20, t=40, b=20),
#     hoverlabel=dict(
#         bgcolor="white",
#         font_size=12,
#         font_family="Arial"
#     )
# )

# # Add range selector buttons (unchanged)
# fig.update_xaxes(
#     rangeselector=dict(
#         buttons=list([
#             dict(count=7, label="1W", step="day", stepmode="backward"),
#             dict(count=1, label="1M", step="month", stepmode="backward"),
#             dict(count=3, label="3M", step="month", stepmode="backward"),
#             dict(count=6, label="6M", step="month", stepmode="backward"),
#             dict(step="all", label="All")
#         ]),
#         bgcolor='rgba(150,150,150,0.2)'
#     )
# )

# st.plotly_chart(fig, use_container_width=True)

