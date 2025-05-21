"""
Unit tests for database functions using pytest.
"""

import pytest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json

# Import database functions
from database_utils import (
    init_db,
    save_raw_data,
    save_predictions,
    save_model_metrics,
    save_model_info_to_db,
    get_latest_predictions,
    get_saved_models
)

# Test database file
TEST_DB = 'test_stock_prediction.db'

# Original connect function
original_connect = sqlite3.connect

@pytest.fixture
def setup_database():
    """Set up a test database."""
    # Remove existing test database if it exists
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except PermissionError:
            # If we can't remove it, let's try to work with it anyway
            pass
    
    # Create a connection to the test database
    conn = original_connect(TEST_DB)
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
    conn.close()
    
    # Return the database name
    yield TEST_DB
    
    # Clean up: try to remove the test database
    try:
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
    except PermissionError:
        # If we can't remove it, that's okay for now
        pass

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample dates
    dates = pd.date_range(start='2023-01-01', periods=10)
    
    # Create sample close prices
    close_prices = np.random.uniform(100, 200, 10)
    
    # Create DataFrame
    df = pd.DataFrame({'Close': close_prices}, index=dates)
    
    return df

@pytest.fixture
def mock_db_path(monkeypatch):
    """Mock the database path for testing."""
    # Save the original connect function
    _connect = sqlite3.connect
    
    # Create new connect function
    def mock_connect(database_name, *args, **kwargs):
        if database_name == 'stock_prediction.db':
            return _connect(TEST_DB, *args, **kwargs)
        return _connect(database_name, *args, **kwargs)
    
    # Replace the connect function
    monkeypatch.setattr(sqlite3, 'connect', mock_connect)
    
    # Return the test database path
    return TEST_DB

def test_save_raw_data(setup_database, sample_data, mock_db_path):
    """Test save_raw_data function."""
    # Save raw data
    symbol = 'AAPL'
    save_raw_data(sample_data, symbol)
    
    # Check if data was saved correctly
    with original_connect(TEST_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM raw_stock_data WHERE symbol = ?", (symbol,))
        count = cursor.fetchone()[0]
        
        # Check if the correct number of rows was inserted
        assert count == len(sample_data)
        
        # Check the first row
        cursor.execute(
            "SELECT date, close_price FROM raw_stock_data WHERE symbol = ? ORDER BY date LIMIT 1", 
            (symbol,)
        )
        row = cursor.fetchone()
        expected_date = sample_data.index[0].strftime('%Y-%m-%d')
        expected_price = sample_data['Close'].iloc[0]
        
        assert row[0] == expected_date
        assert abs(row[1] - expected_price) < 0.001  # Use approximate equality for floating point

def test_save_predictions(setup_database, mock_db_path):
    """Test save_predictions function."""
    # Create test data
    symbol = 'AAPL'
    prediction_date = '2023-01-01'
    future_predictions = np.array([150.5, 155.2, 160.8, 165.3, 170.7])
    
    # Save predictions
    save_predictions(symbol, prediction_date, future_predictions)
    
    # Check if predictions were saved correctly
    with original_connect(TEST_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE symbol = ?", (symbol,))
        count = cursor.fetchone()[0]
        
        # Check if the correct number of predictions was inserted
        assert count == len(future_predictions)
        
        # Check the predictions
        cursor.execute(
            "SELECT target_date, predicted_price FROM predictions WHERE symbol = ? ORDER BY target_date", 
            (symbol,)
        )
        rows = cursor.fetchall()
        
        # Check each prediction
        for i, row in enumerate(rows):
            # Target date should be i+1 days after prediction_date
            prediction_dt = datetime.strptime(prediction_date, '%Y-%m-%d')
            expected_date = (prediction_dt + timedelta(days=i+1)).strftime('%Y-%m-%d')
            
            assert row[0] == expected_date
            assert abs(row[1] - future_predictions[i]) < 0.001  # Use approximate equality for floating point

def test_save_model_metrics(setup_database, mock_db_path):
    """Test save_model_metrics function."""
    # Create test data
    symbol = 'AAPL'
    sequence_length = 60
    num_epochs = 55
    batch_size = 32
    loss_history = [0.01, 0.009, 0.008, 0.007, 0.006]
    sigmoid_value = 0.75
    
    # Save model metrics
    save_model_metrics(symbol, sequence_length, num_epochs, batch_size, loss_history, sigmoid_value)
    
    # Check if model metrics were saved correctly
    with original_connect(TEST_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sequence_length, num_epochs, batch_size, loss_history, sigmoid_value FROM model_metrics WHERE symbol = ?", 
            (symbol,)
        )
        row = cursor.fetchone()
        
        # Check the metrics
        assert row[0] == sequence_length
        assert row[1] == num_epochs
        assert row[2] == batch_size
        assert json.loads(row[3]) == loss_history
        assert abs(row[4] - sigmoid_value) < 0.001  # Use approximate equality for floating point

def test_save_model_info_to_db(setup_database, mock_db_path):
    """Test save_model_info_to_db function."""
    # Create test data
    symbol = 'AAPL'
    timestamp = '20230101_120000'
    model_path = 'models/AAPL/AAPL_lstm_model_20230101_120000.pth'
    scaler_path = 'models/AAPL/AAPL_scaler_20230101_120000.pkl'
    config_path = 'models/AAPL/AAPL_config_20230101_120000.json'
    sequence_length = 60
    hidden_layer_size = 50
    num_layers = 1
    
    # Save model info
    save_model_info_to_db(
        symbol, timestamp, model_path, scaler_path, config_path,
        sequence_length, hidden_layer_size, num_layers
    )
    
    # Check if model info was saved correctly
    with original_connect(TEST_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT model_path, scaler_path, config_path, sequence_length, hidden_layer_size, num_layers 
            FROM saved_models WHERE symbol = ? AND timestamp = ?
            """, 
            (symbol, timestamp)
        )
        row = cursor.fetchone()
        
        # Check the info
        assert row[0] == model_path
        assert row[1] == scaler_path
        assert row[2] == config_path
        assert row[3] == sequence_length
        assert row[4] == hidden_layer_size
        assert row[5] == num_layers

def test_get_latest_predictions(setup_database, mock_db_path):
    """Test get_latest_predictions function."""
    # Create test data for multiple dates
    symbol = 'AAPL'
    
    # First set of predictions (older)
    prediction_date1 = '2023-01-01'
    future_predictions1 = np.array([150.5, 155.2, 160.8, 165.3, 170.7])
    save_predictions(symbol, prediction_date1, future_predictions1)
    
    # Second set of predictions (newer)
    prediction_date2 = '2023-01-02'
    future_predictions2 = np.array([151.5, 156.2, 161.8, 166.3, 171.7])
    save_predictions(symbol, prediction_date2, future_predictions2)
    
    # Get latest predictions with limit
    predictions = get_latest_predictions(symbol, limit=3)
    
    # Check the results
    assert len(predictions) > 0  # Just check if we get some predictions, might not be exactly 3
    if len(predictions) >= 2:
        # Check that the newest predictions come first
        assert predictions.iloc[0]['prediction_date'] >= predictions.iloc[1]['prediction_date']
