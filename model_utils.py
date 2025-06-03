import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
import os
import logging
from datetime import datetime
import math

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

def create_sequences(data, sequence_length):
    """
    Cria sequências temporais para treinamento do LSTM.
    
    Args:
        data (array): Dados normalizados
        sequence_length (int): Comprimento da sequência
    
    Returns:
        tuple: (X, y) onde X são as sequências de entrada e y os valores target
    """
    try:
        if len(data) <= sequence_length:
            logger.error(f"Dados insuficientes: {len(data)} pontos, necessário pelo menos {sequence_length + 1}")
            return None, None
        
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Sequências criadas: X.shape={X.shape}, y.shape={y.shape}")
        return X, y
        
    except Exception as e:
        logger.error(f"Erro ao criar sequências: {e}")
        return None, None

def build_lstm_model(sequence_length, hidden_size=50, num_layers=2):
    """
    Constrói o modelo LSTM.
    
    Args:
        sequence_length (int): Comprimento da sequência
        hidden_size (int): Tamanho da camada oculta
        num_layers (int): Número de camadas LSTM
    
    Returns:
        LSTMModel: Modelo LSTM inicializado
    """
    try:
        model = LSTMModel(
            input_size=1,
            hidden_layer_size=hidden_size,
            num_layers=num_layers,
            output_size=1
        )
        logger.info(f"Modelo LSTM criado: hidden_size={hidden_size}, num_layers={num_layers}")
        return model
        
    except Exception as e:
        logger.error(f"Erro ao construir modelo LSTM: {e}")
        return None

def train_lstm(model, X, y, num_epochs=50, batch_size=32, learning_rate=0.001):
    """
    Treina o modelo LSTM.
    
    Args:
        model (LSTMModel): Modelo LSTM
        X (array): Dados de entrada
        y (array): Dados target
        num_epochs (int): Número de épocas
        batch_size (int): Tamanho do batch
        learning_rate (float): Taxa de aprendizado
    
    Returns:
        tuple: (modelo treinado, histórico de loss, valor sigmoid)
    """
    try:
        # Validação das dimensões dos dados
        if X is None or y is None:
            logger.error("Dados de entrada são None")
            return model, [], 0.5
        
        if len(X.shape) != 3:
            logger.error(f"X deve ter 3 dimensões, mas tem {len(X.shape)}: {X.shape}")
            return model, [], 0.5
        
        if len(y.shape) != 2:
            logger.error(f"y deve ter 2 dimensões, mas tem {len(y.shape)}: {y.shape}")
            return model, [], 0.5
        
        # Converte para tensores do PyTorch
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Define otimizador e função de perda
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        model.train()
        loss_history = []
        
        logger.info(f"Iniciando treinamento: {num_epochs} épocas, batch_size={batch_size}")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Treinamento por batches
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            loss_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Época [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
        
        # Calcula valor sigmoid baseado na perda média
        avg_loss = np.mean(loss_history) if loss_history else 1.0
        sigmoid_value = 1 / (1 + math.exp(avg_loss))
        
        logger.info(f"Treinamento concluído. Loss final: {loss_history[-1]:.6f}, Sigmoid: {sigmoid_value:.4f}")
        
        return model, loss_history, sigmoid_value
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        return model, [], 0.5

def predict_prices(df_scaled, model, scaler, sequence_length, days_ahead=5):
    """
    Faz predições de preços futuros.
    
    Args:
        df_scaled (array): Dados normalizados
        model (LSTMModel): Modelo treinado
        scaler (MinMaxScaler): Scaler usado na normalização
        sequence_length (int): Comprimento da sequência
        days_ahead (int): Número de dias para predizer
    
    Returns:
        tuple: (predição próximo dia, predições futuras desnormalizadas)
    """
    try:
        model.eval()
        
        # Pega os últimos dados para predição
        last_sequence = df_scaled[-sequence_length:].reshape(1, sequence_length, 1)
        last_sequence_tensor = torch.FloatTensor(last_sequence)
        
        # Predição do próximo dia
        with torch.no_grad():
            next_day_pred = model(last_sequence_tensor).item()
        
        # Desnormaliza a predição do próximo dia
        next_day_pred_inv = scaler.inverse_transform([[next_day_pred]])[0][0]
        
        # Predições para múltiplos dias
        future_predictions = []
        current_sequence = df_scaled[-sequence_length:].copy()
        
        for _ in range(days_ahead):
            # Reshapes para o formato correto
            input_seq = current_sequence.reshape(1, sequence_length, 1)
            input_tensor = torch.FloatTensor(input_seq)
            
            with torch.no_grad():
                pred = model(input_tensor).item()
            
            future_predictions.append(pred)
            
            # Atualiza a sequência removendo o primeiro elemento e adicionando a predição
            current_sequence = np.append(current_sequence[1:], pred)
        
        # Desnormaliza as predições futuras
        future_predictions_inv = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        ).flatten()
        
        logger.info(f"Predições geradas: próximo dia = {next_day_pred_inv:.2f}")
        
        return next_day_pred_inv, future_predictions_inv
        
    except Exception as e:
        logger.error(f"Erro ao fazer predições: {e}")
        return 0.0, [0.0] * days_ahead

def save_model(model, scaler, symbol, sequence_length):
    """
    Salva o modelo, scaler e configurações em arquivos.
    
    Args:
        model (LSTMModel): Modelo treinado
        scaler (MinMaxScaler): Scaler usado
        symbol (str): Símbolo da ação
        sequence_length (int): Comprimento da sequência
    
    Returns:
        tuple: (caminho do modelo, caminho do scaler, caminho da config)
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Caminhos dos arquivos
        model_path = f'models/{symbol}_lstm_model_{timestamp}.pth'
        scaler_path = f'models/{symbol}_scaler_{timestamp}.pkl'
        config_path = f'models/{symbol}_config_{timestamp}.json'
        
        # Salva o modelo PyTorch
        torch.save(model.state_dict(), model_path)
        
        # Salva o scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Salva a configuração
        config = {
            'symbol': symbol,
            'sequence_length': sequence_length,
            'hidden_layer_size': model.hidden_layer_size,
            'num_layers': model.num_layers,
            'timestamp': timestamp
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Modelo salvo: {model_path}")
        return model_path, scaler_path, config_path
        
    except Exception as e:
        logger.error(f"Erro ao salvar modelo: {e}")
        return None, None, None

def get_model_summary(model):
    """
    Retorna um resumo do modelo LSTM.
    
    Args:
        model (LSTMModel): Modelo LSTM
    
    Returns:
        dict: Resumo do modelo
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            'hidden_layer_size': model.hidden_layer_size,
            'num_layers': model.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Aproximado em MB
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Erro ao gerar resumo do modelo: {e}")
        return {}

def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo com dados de teste.
    
    Args:
        model (LSTMModel): Modelo treinado
        X_test (array): Dados de teste (entrada)
        y_test (array): Dados de teste (target)
    
    Returns:
        dict: Métricas de avaliação
    """
    try:
        model.eval()
        
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        with torch.no_grad():
            predictions = model(X_test_tensor)
            
            # Calcula métricas
            mse = nn.MSELoss()(predictions, y_test_tensor).item()
            mae = torch.mean(torch.abs(predictions - y_test_tensor)).item()
            rmse = math.sqrt(mse)
            
            # Calcula R²
            y_mean = torch.mean(y_test_tensor)
            ss_tot = torch.sum((y_test_tensor - y_mean) ** 2)
            ss_res = torch.sum((y_test_tensor - predictions) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2.item()
            }
            
            logger.info(f"Avaliação concluída: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.4f}")
            
            return metrics
        
    except Exception as e:
        logger.error(f"Erro durante avaliação: {e}")
        return {}

def load_model(symbol):
    """
    Carrega o modelo mais recente para um símbolo.
    
    Args:
        symbol (str): Símbolo da ação
    
    Returns:
        tuple: (modelo, scaler, config) ou (None, None, None) se não encontrado
    """
    try:
        models_dir = 'models'
        if not os.path.exists(models_dir):
            logger.warning("Diretório de modelos não existe")
            return None, None, None
        
        # Encontra os arquivos mais recentes para o símbolo
        model_files = [f for f in os.listdir(models_dir) 
                      if f.startswith(f'{symbol}_lstm_model_') and f.endswith('.pth')]
        
        if not model_files:
            logger.warning(f"Nenhum modelo encontrado para {symbol}")
            return None, None, None
        
        # Pega o arquivo mais recente
        latest_model_file = sorted(model_files)[-1]
        timestamp = latest_model_file.split('_')[-1].replace('.pth', '')
        
        model_path = os.path.join(models_dir, latest_model_file)
        scaler_path = os.path.join(models_dir, f'{symbol}_scaler_{timestamp}.pkl')
        config_path = os.path.join(models_dir, f'{symbol}_config_{timestamp}.json')
        
        # Carrega a configuração
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Cria o modelo com a configuração
        model = LSTMModel(
            input_size=1,
            hidden_layer_size=config['hidden_layer_size'],
            num_layers=config['num_layers'],
            output_size=1
        )
        
        # Carrega os pesos do modelo
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Carrega o scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        logger.info(f"Modelo carregado: {model_path}")
        return model, scaler, config
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        return None, None, None
