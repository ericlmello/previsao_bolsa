"""
Database functions module for the stock prediction system.
This module contains all the necessary functions to initialize the database,
save and retrieve price data, predictions, and model metrics.
"""

import sqlite3
import logging
import pandas as pd
from datetime import datetime
import json

# Logging setup
logger = logging.getLogger(__name__)

def init_db():
    """
    Initialize SQLite database with required tables.
    
    Creates the following tables if they don't exist:
    - raw_stock_data: raw stock price data
    - predictions: predictions made by the model
    - model_metrics: model performance metrics
    - saved_models: information about saved models
    """
    with sqlite3.connect('stock_prediction.db') as conn:
        cursor = conn.cursor()
        
        # Table for raw stock data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            close_price REAL NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(symbol, date)
        )
        ''')
        
        # Table for model predictions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            target_date TEXT NOT NULL,
            predicted_price REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        ''')

        # Table for model metrics
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            run_date TEXT NOT NULL,
            sequence_length INTEGER NOT NULL,
            num_epochs INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            loss_history TEXT NOT NULL,
            sigmoid_value REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        ''')
        
        # Table for saved models
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            model_path TEXT NOT NULL,
            scaler_path TEXT NOT NULL,
            config_path TEXT NOT NULL,
            sequence_length INTEGER NOT NULL,
            hidden_layer_size INTEGER NOT NULL,
            num_layers INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")

def save_raw_data(df, symbol):
    """Save raw stock data to SQLite database."""
    try:
        with sqlite3.connect('stock_prediction.db') as conn:
            # Reset index to make the date accessible as a column
            df_to_save = df.reset_index()
            
            # Certifique-se que a coluna do índice seja renomeada para 'date'
            # Verifique o nome da coluna do índice após reset_index
            if 'index' in df_to_save.columns:
                df_to_save.rename(columns={'index': 'date'}, inplace=True)
            elif 'Date' in df_to_save.columns:
                df_to_save.rename(columns={'Date': 'date'}, inplace=True)
            
            # Verifique se a coluna 'date' existe após o renomeamento
            if 'date' not in df_to_save.columns:
                # Se não existir, adicione uma coluna 'date' com os valores atuais
                df_to_save['date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Garanta que 'date' seja convertido para string em formato YYYY-MM-DD
            if pd.api.types.is_datetime64_any_dtype(df_to_save['date']):
                df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
            
            # Renomeie 'Close' para 'close_price'
            if 'Close' in df_to_save.columns:
                df_to_save.rename(columns={'Close': 'close_price'}, inplace=True)
            
            # Adicione colunas de símbolo e created_at
            df_to_save['symbol'] = symbol
            df_to_save['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Selecione apenas as colunas necessárias
            df_to_save = df_to_save[['symbol', 'date', 'close_price', 'created_at']]
            
            # Salve no banco de dados
            df_to_save.to_sql('raw_stock_data', conn, if_exists='append', index=False)
            
            logger.info(f"Saved {len(df_to_save)} rows of raw data for {symbol}")
    except Exception as e:
        logger.error(f"Error saving raw data to database: {e}")

def save_predictions(symbol, prediction_date, future_predictions_inv):
    """
    Save predictions to SQLite database.
    
    Args:
        symbol: Stock symbol
        prediction_date: Date of prediction (format 'YYYY-MM-DD')
        future_predictions_inv: Array of predictions for future days
    """
    try:
        with sqlite3.connect('stock_prediction.db') as conn:
            cursor = conn.cursor()
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate target dates (assuming prediction_date is the current date and predictions are for future days)
            prediction_dt = datetime.strptime(prediction_date, '%Y-%m-%d')
            
            # Save each prediction
            for i, pred_price in enumerate(future_predictions_inv):
                # Target date is i+1 days from prediction date
                target_date = (prediction_dt + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d')
                
                cursor.execute('''
                INSERT INTO predictions (symbol, prediction_date, target_date, predicted_price, created_at)
                VALUES (?, ?, ?, ?, ?)
                ''', (symbol, prediction_date, target_date, float(pred_price), created_at))
            
            conn.commit()
            logger.info(f"Saved {len(future_predictions_inv)} predictions for {symbol}")
    except Exception as e:
        logger.error(f"Error saving predictions to database: {e}")

def save_model_metrics(symbol, sequence_length, num_epochs, batch_size, loss_history, sigmoid_value):
    """
    Save model metrics to SQLite database.
    
    Args:
        symbol: Stock symbol
        sequence_length: Sequence length used in the model
        num_epochs: Number of training epochs
        batch_size: Batch size used in training
        loss_history: List of loss values during training
        sigmoid_value: Sigmoid value calculated from average loss
    """
    try:
        with sqlite3.connect('stock_prediction.db') as conn:
            cursor = conn.cursor()
            run_date = datetime.now().strftime('%Y-%m-%d')
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Convert loss_history to JSON string
            loss_history_json = json.dumps(loss_history)
            
            cursor.execute('''
            INSERT INTO model_metrics (symbol, run_date, sequence_length, num_epochs, batch_size, loss_history, sigmoid_value, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, run_date, sequence_length, num_epochs, batch_size, loss_history_json, sigmoid_value, created_at))
            
            conn.commit()
            logger.info(f"Saved model metrics for {symbol}")
    except Exception as e:
        logger.error(f"Error saving model metrics to database: {e}")

def save_model_info_to_db(symbol, timestamp, model_path, scaler_path, config_path, sequence_length, hidden_layer_size, num_layers):
    """
    Save model information to SQLite database.
    
    Args:
        symbol: Stock symbol
        timestamp: Timestamp of the saved model
        model_path: Path to the model file
        scaler_path: Path to the scaler file
        config_path: Path to the configuration file
        sequence_length: Sequence length of the model
        hidden_layer_size: Hidden layer size of the model
        num_layers: Number of layers in the model
    """
    try:
        with sqlite3.connect('stock_prediction.db') as conn:
            cursor = conn.cursor()
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute('''
            INSERT INTO saved_models (symbol, timestamp, model_path, scaler_path, config_path, 
                                     sequence_length, hidden_layer_size, num_layers, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, timestamp, model_path, scaler_path, config_path, 
                 sequence_length, hidden_layer_size, num_layers, created_at))
            
            conn.commit()
            logger.info(f"Saved model information to database for {symbol}")
    except Exception as e:
        logger.error(f"Error saving model information to database: {e}")

def get_latest_predictions(symbol, limit=5):
    """
    Get the latest predictions from the database.
    
    Args:
        symbol: Stock symbol
        limit: Number of predictions to return (default = 5)
        
    Returns:
        DataFrame with the latest predictions
    """
    try:
        with sqlite3.connect('stock_prediction.db') as conn:
            query = '''
            SELECT prediction_date, target_date, predicted_price
            FROM predictions
            WHERE symbol = ?
            ORDER BY id DESC
            LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(symbol, limit))
            return df
    except Exception as e:
        logger.error(f"Error getting latest predictions: {e}")
        return pd.DataFrame()

def get_saved_models(symbol):
    """
    Get the list of saved models for a specific symbol.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        DataFrame with information about saved models
    """
    try:
        with sqlite3.connect('stock_prediction.db') as conn:
            query = '''
            SELECT symbol, timestamp, model_path, sequence_length, created_at
            FROM saved_models
            WHERE symbol = ?
            ORDER BY id DESC
            '''
            df = pd.read_sql_query(query, conn, params=(symbol,))
            return df
    except Exception as e:
        logger.error(f"Error getting saved models: {e}")
        return pd.DataFrame()
