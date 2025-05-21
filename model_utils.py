import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
import pickle
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Function to create sequences (NOVO!)
def create_sequences(data, seq_length):
    """
    Create input sequences and target values for time series prediction.
    
    Args:
        data: Normalized input data (numpy array)
        seq_length: Length of input sequences
        
    Returns:
        X: Input sequences
        y: Target values
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def build_lstm_model(sequence_length):
    input_size = 1
    hidden_size = 50
    output_size = 1
    num_layers = 1
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    logger.info(f"LSTM Model created with sequence_length={sequence_length}, hidden_size={hidden_size}")
    return model

def train_lstm(model, X, y, num_epochs=55, batch_size=32, mlflow_available=False, prometheus_available=False, MODEL_LOSS=None, MODEL_SIGMOID=None):
    X_tensor = torch.FloatTensor(X.reshape(-1, X.shape[1], 1))
    y_tensor = torch.FloatTensor(y)
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        if mlflow_available:
            import mlflow
            mlflow.log_metric("loss", avg_loss, step=epoch)
        if prometheus_available and MODEL_LOSS:
            MODEL_LOSS.observe(avg_loss)
        if epoch % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    avg_overall_loss = np.mean(loss_history)
    sigmoid_value = 1 / (1 + np.exp(-avg_overall_loss))
    if prometheus_available and MODEL_SIGMOID:
        MODEL_SIGMOID.labels(model='pytorch_lstm').set(sigmoid_value)
    return model, loss_history, sigmoid_value

def predict_prices(data, model, scaler, sequence_length, days_ahead=1):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    current_sequence = data[-sequence_length:].reshape(1, sequence_length, 1)
    current_sequence = torch.FloatTensor(current_sequence).to(device)
    predictions = []
    for _ in range(days_ahead):
        with torch.no_grad():
            prediction = model(current_sequence)
        predicted_value = prediction.item()
        predictions.append(predicted_value)
        new_sequence = torch.cat((current_sequence[:, 1:, :], prediction.view(1, 1, 1)), dim=1)
        current_sequence = new_sequence
    predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions_inv[0], predictions_inv

# Add save_model function if it doesn't exist
def save_model(model, scaler, symbol, sequence_length):
    """Save the trained model to files."""
    try:
        # Create timestamp for model versioning
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create model directory if it doesn't exist
        model_dir = os.path.join('models', symbol)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save model
        model_path = os.path.join(model_dir, f"{symbol}_lstm_model_{timestamp}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'sequence_length': sequence_length,
            'hidden_layer_size': model.hidden_layer_size,
            'num_layers': model.num_layers,
            'timestamp': timestamp
        }, model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, f"{symbol}_scaler_{timestamp}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save a config file with metadata
        config_path = os.path.join(model_dir, f"{symbol}_config_{timestamp}.json")
        config = {
            'symbol': symbol,
            'sequence_length': sequence_length,
            'hidden_layer_size': model.hidden_layer_size,
            'num_layers': model.num_layers,
            'timestamp': timestamp,
            'model_path': model_path,
            'scaler_path': scaler_path
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Also save as latest model for easy access
        latest_model_path = os.path.join(model_dir, f"{symbol}_lstm_model_latest.pth")
        latest_scaler_path = os.path.join(model_dir, f"{symbol}_scaler_latest.pkl")
        latest_config_path = os.path.join(model_dir, f"{symbol}_config_latest.json")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'sequence_length': sequence_length,
            'hidden_layer_size': model.hidden_layer_size,
            'num_layers': model.num_layers,
            'timestamp': timestamp
        }, latest_model_path)
        
        with open(latest_scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(latest_config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Model saved to {model_path} and {latest_model_path}")
        logger.info(f"Scaler saved to {scaler_path} and {latest_scaler_path}")
        logger.info(f"Config saved to {config_path} and {latest_config_path}")
        
        return model_path, scaler_path, config_path
    
    except Exception as e:
        logger.error(f"Error saving model to file: {e}")
        return None, None, None

# Add load_model function if it doesn't exist
def load_model(symbol, use_latest=True, timestamp=None):
    """Load a trained model from files."""
    try:
        model_dir = os.path.join('models', symbol)
        
        if use_latest:
            model_path = os.path.join(model_dir, f"{symbol}_lstm_model_latest.pth")
            scaler_path = os.path.join(model_dir, f"{symbol}_scaler_latest.pkl")
            config_path = os.path.join(model_dir, f"{symbol}_config_latest.json")
        else:
            if timestamp is None:
                logger.error("Timestamp must be provided when use_latest is False")
                return None, None, None
            
            model_path = os.path.join(model_dir, f"{symbol}_lstm_model_{timestamp}.pth")
            scaler_path = os.path.join(model_dir, f"{symbol}_scaler_{timestamp}.pkl")
            config_path = os.path.join(model_dir, f"{symbol}_config_{timestamp}.json")
        
        # Check if files exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(config_path):
            logger.error(f"Model, scaler, or config file not found")
            return None, None, None
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load model
        checkpoint = torch.load(model_path)
        sequence_length = checkpoint['sequence_length']
        hidden_layer_size = checkpoint['hidden_layer_size']
        num_layers = checkpoint['num_layers']
        
        # Create model with same architecture
        model = LSTMModel(
            input_size=1,
            hidden_layer_size=hidden_layer_size,
            output_size=1,
            num_layers=num_layers
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Scaler loaded from {scaler_path}")
        logger.info(f"Config loaded from {config_path}")
        
        return model, scaler, config
    
    except Exception as e:
        logger.error(f"Error loading model from file: {e}")
        return None, None, None
