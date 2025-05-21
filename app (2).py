import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import logging
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.pytorch
import time
import ssl
from yfinance_utils import download_stock_data

# Import database functions (with English names)
from database_utils import (
    init_db,
    save_raw_data,
    save_predictions,
    save_model_metrics,
    save_model_info_to_db,
    get_latest_predictions,
    get_saved_models
)

# Import model functions
from model_utils import (
    create_sequences,
    build_lstm_model,
    train_lstm,
    predict_prices,
    save_model,
    load_model
)

# Try to import Prometheus libraries, but continue if they fail
try:
    from prometheus_flask_exporter import PrometheusMetrics
    from prometheus_client import Summary, Gauge
    prometheus_available = True
except ImportError:
    prometheus_available = False
    print("Prometheus unavailable. Metrics will not be collected.")

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyTorch imports
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("PyTorch unavailable. LSTM model will not work.")

# MLflow imports
try:
    import mlflow
    import mlflow.pytorch
    mlflow_available = True
except ImportError:
    mlflow_available = False
    print("MLflow unavailable. Experiment tracking will not work.")

# Creating the static directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Flask configuration
app = Flask(__name__)

# Initialize Prometheus
if prometheus_available:
    metrics = PrometheusMetrics(app)
    metrics.info('appinfos', 'Application info', version='1.0.0')
    MODEL_LOSS = Summary('model_loss_count', 'Loss function during training')
    MODEL_ACCURACY = Summary('model_accuracy_count', 'Accuracy during training')
    MODEL_TRAINING_TIME = Summary('model_training_duration_seconds', 'Time spent training the model')
    MODEL_SIGMOID = Gauge('model_sigmoid_value', 'Sigmoid value computed from the average training loss', ['model'])

# Function to download data
def download_with_timeout(symbol, start_date, end_date, timeout=180):
    try:
        logging.info(f"Downloading data for symbol {symbol} from {start_date} to {end_date}.")
        
        # Use a nova função com retry e cache
        df = download_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            max_retries=3,  # Tente até 3 vezes
            base_delay=2,   # Começando com 2 segundos de atraso
            use_cache=True  # Use cache para reduzir requisições
        )
        
        if df.empty:
            logging.warning(f"No data found for symbol {symbol} between {start_date} and {end_date}.")
        return df
    except Exception as e:
        logging.error(f"Error collecting data: {e}")
        return None

#------------------------------------------------------------
# FLASK ROUTES
#------------------------------------------------------------

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/history')
def history():
    """Page to view historical predictions."""
    symbol = request.args.get('symbol', 'QQQ')
    latest_predictions = get_latest_predictions(symbol)
    return render_template('history.html', predictions=latest_predictions, symbol=symbol)

@app.route('/models')
def list_models():
    """Page to view saved models."""
    symbol = request.args.get('symbol', 'QQQ')
    models_df = get_saved_models(symbol)
    return render_template('models.html', models=models_df, symbol=symbol)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        symbol = request.form.get('symbol', 'QQQ')  
        use_saved_model = request.form.get('use_saved_model') == 'on'
        
        logging.info(f"Receiving data for symbol {symbol} from {start_date} to {end_date}.")
        logging.info(f"Use saved model: {use_saved_model}")
        
        # Download data using yfinance
        df = download_with_timeout(symbol, start_date, end_date)
        if df is None or df.empty:
            logging.error("Error collecting data. Check the dates and try again.")
            return "Error collecting data. Check the dates and try again."
        
        # Select only the 'Close' column
        df_close = df[['Close']]
        logging.info("Data downloaded successfully.")

        # Save raw data to SQLite
        save_raw_data(df_close, symbol)

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df_close)
        
        # Create time sequences
        sequence_length = 60
        X, y = create_sequences(df_scaled, sequence_length)
        
        # Set the number of epochs for training
        num_epochs = 55  
        batch_size = 32 
        
        if use_saved_model:
            # Try to load the latest saved model
            model, saved_scaler, config = load_model(symbol)
            
            if model is not None and saved_scaler is not None:
                logger.info(f"Using saved model for {symbol}")
                # Use the saved scaler for new data
                scaler = saved_scaler
                # Re-normalize data with the saved scaler
                df_scaled = scaler.transform(df_close)
            else:
                logger.warning(f"No saved model found for {symbol}. Training a new model.")
                use_saved_model = False
        
        if not use_saved_model:
            # Create and train a new model
            model = build_lstm_model(sequence_length)
            
            # Train the model with Prometheus monitoring
            if prometheus_available:
                # Start Prometheus time tracking
                with MODEL_TRAINING_TIME.time():
                    # Train model with or without MLflow
                    if mlflow_available:
                        mlflow.set_experiment("LSTM Stock Prediction PyTorch")
                        with mlflow.start_run():
                            mlflow.log_param("sequence_length", sequence_length)
                            mlflow.log_param("num_epochs", num_epochs)
                            mlflow.log_param("batch_size", batch_size)
                            
                            # Train the model with MLflow tracking
                            model, loss_history, sigmoid_value = train_lstm(model, X, y, num_epochs, batch_size)
                            
                            # Log the PyTorch model
                            mlflow.pytorch.log_model(model, "lstm_model")
                    else:
                        # Train without MLflow
                        model, loss_history, sigmoid_value = train_lstm(model, X, y, num_epochs, batch_size)
            else:
                # No Prometheus monitoring
                if mlflow_available:
                    mlflow.set_experiment("LSTM Stock Prediction PyTorch")
                    with mlflow.start_run():
                        mlflow.log_param("sequence_length", sequence_length)
                        mlflow.log_param("num_epochs", num_epochs)
                        mlflow.log_param("batch_size", batch_size)
                        
                        # Train the model with MLflow tracking
                        model, loss_history, sigmoid_value = train_lstm(model, X, y, num_epochs, batch_size)
                        
                        # Log the PyTorch model
                        mlflow.pytorch.log_model(model, "lstm_model")
                else:
                    # Train without MLflow and Prometheus
                    model, loss_history, sigmoid_value = train_lstm(model, X, y, num_epochs, batch_size)

            # Save model metrics to database
            save_model_metrics(symbol, sequence_length, num_epochs, batch_size, loss_history, sigmoid_value)
            
            # Save the trained model to file
            model_path, scaler_path, config_path = save_model(model, scaler, symbol, sequence_length)
            
            if model_path is not None:
                # Save model information to database
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_model_info_to_db(
                    symbol, 
                    timestamp, 
                    model_path, 
                    scaler_path, 
                    config_path, 
                    sequence_length, 
                    model.hidden_layer_size, 
                    model.num_layers
                )

        # Predict next day and next 5 days
        prediction_next_day, future_predictions_inv = predict_prices(df_scaled, model, scaler, sequence_length, days_ahead=5)

        # Save predictions to database
        current_date = datetime.now().strftime('%Y-%m-%d')
        save_predictions(symbol, current_date, future_predictions_inv)

        logging.info(f"Next day prediction: {prediction_next_day}")
        
        # Calculate percentage change for the next day
        last_close = df_close['Close'].iloc[-1]
        price_change = prediction_next_day - last_close
        percent_change = (price_change / last_close) * 100
        
        # Create dictionary with change information
        changes = {
            'last_close': round(last_close, 2),
            'price_change': round(price_change, 2),
            'percent_change': round(percent_change, 2)
        }
        
        # Chart 1: Closing Price Over Time
        line_chart_path = 'static/line_chart.png'
        plt.figure(figsize=(12, 6))
        plt.plot(df_close.index, df_close['Close'], label='Closing Price')
        plt.title('Stock Closing Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(line_chart_path)
        plt.close()

        # Chart 2: Histogram of price distribution
        hist_chart_path = 'static/hist_chart.png'
        plt.figure(figsize=(10, 6))
        plt.hist(df_close['Close'], bins=50, color='blue', edgecolor='black')
        plt.title('Closing Price Distribution')
        plt.xlabel('Closing Price')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(hist_chart_path)
        plt.close()

        # Chart 3: Next 5 Days Prediction
        future_chart_path = 'static/future_chart.png'
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 6), future_predictions_inv, marker='o', linestyle='-', label='Prediction')
        plt.title('Next 5 Days Prediction')
        plt.xlabel('Day')
        plt.ylabel('Predicted Price')
        plt.grid(True)
        plt.legend()
        plt.savefig(future_chart_path)
        plt.close()

        # If we trained a new model, show the learning curve
        if not use_saved_model:
            # Chart 4: Learning curve (Loss over epochs)
            learning_curve_path = 'static/learning_curve.png'
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, num_epochs+1), loss_history, label='Loss (MSE)')
            plt.title('Learning Curve (Loss)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.savefig(learning_curve_path)
            plt.close()

            # Chart 5: Sigmoid Metric
            sigmoid_chart_path = 'static/sigmoid_chart.png'
            plt.figure(figsize=(10, 6))
            plt.axhline(y=sigmoid_value, color='r', linestyle='--', label=f'Sigmoid Value: {sigmoid_value:.4f}')
            plt.title('Sigmoid Function Value (based on average loss)')
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.ylim([0, 1])
            plt.legend()
            plt.grid(True)
            plt.savefig(sigmoid_chart_path)
            plt.close()

            return render_template('result.html',
                                prediction_next_day=round(prediction_next_day, 2),
                                line_chart=line_chart_path,
                                hist_chart=hist_chart_path,
                                future_chart=future_chart_path,
                                learning_curve=learning_curve_path,
                                sigmoid_chart=sigmoid_chart_path,
                                changes=changes,
                                symbol=symbol,
                                model_saved=True)
        else:
            # For saved model, we don't have learning curves
            return render_template('result.html',
                                prediction_next_day=round(prediction_next_day, 2),
                                line_chart=line_chart_path,
                                hist_chart=hist_chart_path,
                                future_chart=future_chart_path,
                                changes=changes,
                                symbol=symbol,
                                model_saved=False,
                                used_saved_model=True)
    
    return render_template('form.html')

@app.route('/train_model', methods=['GET', 'POST'])
def train_model_route():
    """Route to train and save a model without making predictions."""
    if request.method == 'POST':
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        symbol = request.form.get('symbol', 'QQQ')
        num_epochs = int(request.form.get('num_epochs', 55))
        sequence_length = int(request.form.get('sequence_length', 60))
        
        logging.info(f"Training model for symbol {symbol} with data from {start_date} to {end_date}.")
        
        # Download data using yfinance
        df = download_with_timeout(symbol, start_date, end_date)
        if df is None or df.empty:
            logging.error("Error collecting data. Check the dates and try again.")
            return "Error collecting data. Check the dates and try again."
        
        # Select only the 'Close' column
        df_close = df[['Close']]
        logging.info("Data downloaded successfully.")
        
        # Save raw data to SQLite
        save_raw_data(df_close, symbol)
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df_close)
        
        # Create time sequences
        X, y = create_sequences(df_scaled, sequence_length)
        
        # Create the model
        model = build_lstm_model(sequence_length)
        
        # Train the model
        batch_size = 32
        model, loss_history, sigmoid_value = train_lstm(model, X, y, num_epochs, batch_size)
        
        # Save model metrics to database
        save_model_metrics(symbol, sequence_length, num_epochs, batch_size, loss_history, sigmoid_value)
        
        # Save the trained model to file
        model_path, scaler_path, config_path = save_model(model, scaler, symbol, sequence_length)
        
        if model_path is not None:
            # Save model information to database
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_model_info_to_db(
                symbol, 
                timestamp, 
                model_path, 
                scaler_path, 
                config_path, 
                sequence_length, 
                model.hidden_layer_size, 
                model.num_layers
            )
            
            return f"Model trained and saved successfully to {model_path}"
        else:
            return "Error saving the trained model."
    
    return render_template('train_model.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=False, host='0.0.0.0')
