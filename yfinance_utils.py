"""
Utility functions for Yahoo Finance API interactions
with rate limiting, caching mechanisms, and fallback data.
"""

import logging
import time
import os
import pickle
import json
from datetime import datetime, timedelta
import random
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import ssl

# Setup logging
logger = logging.getLogger(__name__)

# Set up cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Default User-Agent
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

def get_cache_path(symbol, start_date, end_date):
    """Generate a cache file path for stock data."""
    cache_filename = f"{symbol}_{start_date}_{end_date}.pkl"
    return os.path.join(CACHE_DIR, cache_filename)

def save_to_cache(data, symbol, start_date, end_date):
    """Save stock data to cache file."""
    try:
        cache_path = get_cache_path(symbol, start_date, end_date)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to cache: {cache_path}")
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")

def load_from_cache(symbol, start_date, end_date, cache_expiration_days=7):  # Increased cache validity
    """Load stock data from cache if available and not expired."""
    try:
        cache_path = get_cache_path(symbol, start_date, end_date)
        
        # Check if cache exists
        if not os.path.exists(cache_path):
            return None
        
        # Check if cache is expired
        file_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        expiration_time = datetime.now() - timedelta(days=cache_expiration_days)
        
        if file_modified_time < expiration_time:
            logger.info(f"Cache expired for {symbol} from {start_date} to {end_date}")
            return None
        
        # Load from cache
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Data loaded from cache: {cache_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading from cache: {e}")
        return None

def generate_sample_data(symbol, start_date, end_date):
    """
    Generate sample stock data when API fails.
    
    Args:
        symbol: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with sample stock data
    """
    try:
        # Parse dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate date range (business days)
        dates = pd.date_range(start=start, end=end, freq='B')
        
        # Base price and daily volatility based on the symbol
        # Use a hash of the symbol name for consistency
        symbol_hash = sum(ord(c) for c in symbol)
        base_price = 100 + (symbol_hash % 400)  # Base price between 100 and 500
        volatility = 0.01 + (symbol_hash % 10) * 0.001  # Volatility between 1-2%
        
        # Generate sample prices with random walk
        np.random.seed(symbol_hash)  # Same seed = same data for same symbol
        returns = np.random.normal(0.0005, volatility, size=len(dates))
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = prices[1:]  # Remove the initial price
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            'Low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            'Close': [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices],
            'Adj Close': [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices],
            'Volume': [int(np.random.uniform(100000, 10000000)) for _ in prices]
        }, index=dates)
        
        logger.warning(f"Using SAMPLE DATA for {symbol}. This is not real market data!")
        return df
    
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        # Return minimal valid DataFrame
        return pd.DataFrame({
            'Open': [100],
            'High': [101],
            'Low': [99],
            'Close': [100.5],
            'Adj Close': [100.5],
            'Volume': [1000000]
        }, index=[pd.to_datetime('today')])

def download_stock_data(symbol, start_date, end_date, max_retries=5, base_delay=5, use_cache=True, use_fallback=True):
    """
    Download stock data with retry mechanism, caching and fallback to sample data.
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        max_retries (int): Maximum number of retry attempts
        base_delay (int): Base delay between retries in seconds
        use_cache (bool): Whether to use caching mechanism
        use_fallback (bool): Whether to use fallback sample data when API fails
        
    Returns:
        pandas.DataFrame: Stock data
    """
    # Check cache first if enabled
    if use_cache:
        cached_data = load_from_cache(symbol, start_date, end_date)
        if cached_data is not None and not cached_data.empty:
            return cached_data
    
    # Fix SSL certificate issues
    if not hasattr(ssl, '_create_default_https_context'):
        ssl._create_default_https_context = ssl._create_unverified_context
    
    logger.info(f"Downloading data for symbol {symbol} from {start_date} to {end_date}")
    
    # Create session with custom headers
    session = requests.Session()
    session.verify = False
    session.headers.update({'User-Agent': DEFAULT_USER_AGENT})
    
    # Try downloading with retries
    for attempt in range(max_retries):
        try:
            # Add some randomization to delay to avoid synchronized requests
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            
            # If not the first attempt, sleep before retrying
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s delay")
                time.sleep(delay)
            
            # Download data
            df = yf.download(
                symbol, 
                start=start_date, 
                end=end_date, 
                progress=False,
                session=session
            )
            
            # Check if data is empty
            if df.empty:
                logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
                if use_fallback:
                    logger.info("Using fallback sample data")
                    df = generate_sample_data(symbol, start_date, end_date)
                else:
                    return pd.DataFrame()
            
            # Save to cache if enabled
            if use_cache:
                save_to_cache(df, symbol, start_date, end_date)
            
            return df
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for rate limit errors
            if "rate limit" in error_msg or "too many requests" in error_msg:
                logger.warning(f"Rate limit exceeded. Attempt {attempt + 1}/{max_retries}")
                # If this was the last attempt and fallback is enabled, use sample data
                if attempt == max_retries - 1 and use_fallback:
                    logger.warning("Rate limit persists. Using fallback sample data.")
                    df = generate_sample_data(symbol, start_date, end_date)
                    
                    # Save to cache
                    if use_cache:
                        save_to_cache(df, symbol, start_date, end_date)
                    
                    return df
                
                # Continue to the next attempt with increased delay
                continue
            
            # For other errors, log and retry
            logger.error(f"Error downloading data (attempt {attempt + 1}/{max_retries}): {e}")
            
            # If this was the last attempt
            if attempt == max_retries - 1:
                if use_fallback:
                    logger.warning("All attempts failed. Using fallback sample data.")
                    df = generate_sample_data(symbol, start_date, end_date)
                    
                    # Save to cache
                    if use_cache:
                        save_to_cache(df, symbol, start_date, end_date)
                    
                    return df
                else:
                    logger.error("Max retries reached, giving up.")
                    raise
    
    # This should not be reached due to the conditional logic above
    return pd.DataFrame()
