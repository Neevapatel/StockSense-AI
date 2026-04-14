import finnhub
from nsetools import Nse
import yfinance as yf
import requests
import os
from dotenv import load_dotenv

load_dotenv() 

# Initialize Finnhub Client
api_key = os.getenv("FINNHUB_API_KEY")
finnhub_client = finnhub.Client(api_key=api_key)

def get_stock_quote(symbol):
    """Fetches real-time price using yfinance with strict validation"""
    try:
        clean_symbol = symbol.strip().upper()
        ticker = yf.Ticker(clean_symbol)
        
        # 1. VALIDATION: fetching 1 day of history. 
        # If the symbol is invalid (e.g., 'ABCXYZ'), this returns an empty DataFrame.
        history = ticker.history(period="1d")
        if history.empty:
            return {
                "success": False, 
                "error": f"Symbol '{clean_symbol}' not found. Try 'AAPL' or 'RELIANCE.NS'."
            }
        
        # 2. DATA EXTRACTION: Using fast_info for performance
        info = ticker.fast_info
        current_price = info['last_price']
        prev_close = info['previous_close']
        
        # Checking for zero/NaN cases which can happen with delisted stocks
        if current_price is None or current_price == 0:
            return {"success": False, "error": "Price data currently unavailable for this symbol."}

        change = current_price - prev_close
        percent_change = (change / prev_close) * 100 if prev_close != 0 else 0

        return {
            "success": True,
            "current": round(current_price, 2),
            "change": round(change, 2),
            "percent": round(percent_change, 2),
            "prev_close": round(prev_close, 2),
            "symbol": clean_symbol
        }
    except Exception as e:
        print(f"YFinance Error: {e}")
        return {"success": False, "error": "Connection error. Please try again later."}


def get_market_news():
    api_key = os.getenv("FINNHUB_API_KEY")
    url = f"https://finnhub.io/api/v1/news?category=general&token={api_key}"
    try:
        response = requests.get(url)
        return response.json()[:5] # Limit to top 5 for the sidebar
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []
    
def get_comparison_data(symbol1, symbol2):
    """Fetches data for two stocks for side-by-side comparison"""
    s1_data = get_stock_quote(symbol1)
    s2_data = get_stock_quote(symbol2)
    
    if not s1_data['success'] or not s2_data['success']:
        return {"success": False, "error": "One or both symbols are invalid."}
        
    return {
        "success": True,
        "stock1": s1_data,
        "stock2": s2_data
    }

def get_stock_history(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # Fetching last 30 days of data at 1-day intervals
        hist = ticker.history(period="1mo", interval="1d")
        if hist.empty:
            return None
        
        # Format data for Chart.js
        return {
            "labels": [date.strftime('%Y-%m-%d') for date in hist.index],
            "prices": [round(price, 2) for price in hist['Close']]
        }
    except Exception as e:
        print(f"History Error: {e}")
        return None
    
def get_trending_stocks():
    # A list of active NSE stocks
    symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ZOMATO.NS', 'TATAMOTORS.NS']
    trending_data = []

    for symbol in symbols:
        try:
            # Fetching 5 days of data to ensure we have enough for a % change calculation
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d")
            
            if not df.empty and len(df) >= 2:
                current_price = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2]
                change = current_price - prev_close
                percent = (change / prev_close) * 100

                trending_data.append({
                    'symbol': symbol,
                    'current': round(current_price, 2),
                    'change': round(change, 2),
                    'percent': round(percent, 2)
                })
        except:
            continue
    
    # Sorting: Putting the biggest gainers at the top
    return sorted(trending_data, key=lambda x: x['percent'], reverse=True)[:5]
    
def get_market_indices():
    indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN"
    }
    results = []
    

    for name, ticker_symbol in indices.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Using period="5d" to ensure we cross weekends/holidays
            data = ticker.history(period="5d")
            
            if not data.empty and len(data) >= 2:
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                
                change = current_price - prev_price
                percent = (change / prev_price) * 100
                
                results.append({
                    "name": name,
                    "price": round(current_price, 2),
                    "percent": round(percent, 2),
                    "change": round(change, 2)
                })
        except Exception as e:
            print(f"Index Error for {name}: {e}")
            continue
            
    return results