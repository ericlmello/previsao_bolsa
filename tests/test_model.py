"""
Unit tests for model functions using pytest.
"""

import pytest
import numpy as np
import os
import pickle
from unittest.mock import patch, MagicMock

# Import model functions
from model_utils import (
    LSTMModel,
    create_sequences,
    build_lstm_model,
    save_model,
    load_model,
    predict_prices
)

# Check if PyTorch is available
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("PyTorch is not available. Skipping PyTorch tests.")

# Skip tests if PyTorch is not available
pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch not available")

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Random data points
    data_points = 100
    
    # Create sample data as a 2D array (like DataFrame['Close'])
    data = np.random.uniform(100, 200, data_points).reshape(-1, 1)
    
    return data

@pytest.fixture
def sample_model():
    """Create a sample LSTM model for testing."""
    if not torch_available:
        return None
    
    model = build_lstm_model(sequence_length=60)
    return model

def test_create_sequences(sample_data):
    """Test create_sequences function."""
    # Parameters
    seq_length = 10
    
    # Create sequences
    X, y = create_sequences(sample_data, seq_length)
    
    # Verify shapes
    assert X.shape[0] == len(sample_data) - seq_length
    assert X.shape[1] == seq_length
    assert y.shape[0] == len(sample_data) - seq_length
    
    # Verify that sequences are correct (modified test)
    for i in range(len(X)):
        # Check that X[i] contains the same values as the slice of sample_data
        for j in range(seq_length):
            assert X[i][j] == sample_data[i+j][0]
        
        # Check that y[i] is the value that follows the sequence
        assert y[i] == sample_data[i+seq_length][0]

def test_build_lstm_model():
    """Test build_lstm_model function."""
    # Build model with default parameters
    model = build_lstm_model(sequence_length=60)
    
    # Verify model structure
    assert isinstance(model, LSTMModel)
    assert model.hidden_layer_size == 50
    assert model.num_layers == 1
    
    # Test forward pass with a sample input
    batch_size = 1
    seq_length = 60
    input_tensor = torch.randn(batch_size, seq_length, 1)
    output = model(input_tensor)
    
    # Output should be a tensor with shape [batch_size, output_size]
    assert output.shape == (batch_size, 1)

@pytest.fixture
def mock_temp_dir(tmpdir):
    """Create a temporary directory for model files."""
    # Create model subdirectory
    model_dir = tmpdir.mkdir("models").mkdir("AAPL")
    return model_dir

@patch('torch.save')
@patch('pickle.dump')
@patch('json.dump')
def test_save_model(mock_json_dump, mock_pickle_dump, mock_torch_save, sample_model, mock_temp_dir):
    """Test save_model function with mocks."""
    # Setup
    model = sample_model
    mock_scaler = MagicMock()
    symbol = 'AAPL'
    sequence_length = 60
    
    # Override os.path.exists to return True for the model directory
    with patch('os.path.exists', return_value=True):
        # Override os.makedirs to do nothing
        with patch('os.makedirs'):
            # Call save_model with mocked file operations
            with patch('builtins.open', MagicMock()):
                model_path, scaler_path, config_path = save_model(model, mock_scaler, symbol, sequence_length)
    
    # Check if torch.save was called at least once
    assert mock_torch_save.call_count >= 1
    
    # Check if pickle.dump was called for scaler
    assert mock_pickle_dump.call_count >= 1
    
    # Check if json.dump was called for config
    assert mock_json_dump.call_count >= 1
    
    # Check that we got valid paths back
    assert model_path is not None
    assert scaler_path is not None
    assert config_path is not None
    
    # Check the returned paths format
    assert symbol in str(model_path)
    assert "lstm_model" in str(model_path)

@patch('torch.load')
@patch('pickle.load')
@patch('json.load')
@patch('os.path.exists', return_value=True)
def test_load_model(mock_exists, mock_json_load, mock_pickle_load, mock_torch_load):
    """Test load_model function with mocks."""
    # Setup mocks
    mock_checkpoint = {
        'model_state_dict': {
            'lstm.weight_ih_l0': torch.randn(200, 1),  # Add mock state dictionary items
            'lstm.weight_hh_l0': torch.randn(200, 50),
            'lstm.bias_ih_l0': torch.randn(200),
            'lstm.bias_hh_l0': torch.randn(200),
            'fc.weight': torch.randn(1, 50),
            'fc.bias': torch.randn(1)
        },
        'sequence_length': 60,
        'hidden_layer_size': 50,
        'num_layers': 1,
        'timestamp': '20230101_120000'
    }
    mock_torch_load.return_value = mock_checkpoint
    
    mock_config = {
        'symbol': 'AAPL',
        'sequence_length': 60,
        'hidden_layer_size': 50,
        'num_layers': 1,
        'timestamp': '20230101_120000',
        'model_path': 'models/AAPL/AAPL_lstm_model_20230101_120000.pth',
        'scaler_path': 'models/AAPL/AAPL_scaler_20230101_120000.pkl'
    }
    mock_json_load.return_value = mock_config
    
    mock_scaler = MagicMock()
    mock_pickle_load.return_value = mock_scaler
    
    # Mock the model's load_state_dict to do nothing
    with patch.object(LSTMModel, 'load_state_dict'):
        # Call load_model with mocks
        with patch('builtins.open', MagicMock()):
            model, scaler, config = load_model('AAPL', use_latest=True)
    
    # Check if the model was loaded
    assert model is not None
    assert isinstance(model, LSTMModel)
    assert model.hidden_layer_size == 50
    assert model.num_layers == 1
    
    # Check if load was called
    assert mock_torch_load.call_count == 1
    assert mock_pickle_load.call_count == 1
    assert mock_json_load.call_count == 1
    
    # Check returned objects
    assert scaler == mock_scaler
    assert config == mock_config

@patch('torch.FloatTensor')
@patch('torch.cat')
def test_predict_prices(mock_cat, mock_float_tensor, sample_model, sample_data):
    """Test predict_prices function with mocks."""
    # Setup mocks
    model = sample_model
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    model.forward = MagicMock(side_effect=[torch.tensor([[0.5]])] * 5)  # Return 0.5 for each prediction
    
    mock_float_tensor.return_value = torch.tensor([[[0.5]]])
    mock_cat.return_value = torch.tensor([[[0.5]]])
    
    # Create a mock scaler
    mock_scaler = MagicMock()
    mock_future_values = np.array([[150.0], [152.0], [154.0], [156.0], [158.0]])
    mock_scaler.inverse_transform.return_value = mock_future_values
    
    # Call predict_prices with mocked torch.no_grad
    mock_context = MagicMock()
    mock_context.__enter__ = MagicMock()
    mock_context.__exit__ = MagicMock()
    
    with patch('torch.no_grad', return_value=mock_context):
        # Make prediction
        next_day, future_predictions = predict_prices(
            sample_data, model, mock_scaler, sequence_length=60, days_ahead=5
        )
    
    # Check the types of the results
    assert isinstance(next_day, float)
    assert isinstance(future_predictions, np.ndarray)
    
    # Check that we got the expected values
    assert next_day == mock_future_values[0][0]
    assert len(future_predictions) == 5
