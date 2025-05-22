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

from database_utils import (
    init_db,
    save_raw_data,
    save_predictions,
    save_model_metrics,
    save_model_info_to_db,
    get_latest_predictions,
    get_saved_models
)

from model_utils import (
    create_sequences,
    build_lstm_model,
    train_lstm,
    predict_prices,
    save_model,
    load_model
)

try:
    from prometheus_flask_exporter import PrometheusMetrics
    from prometheus_client import Summary, Gauge
    prometheus_available = True
except ImportError:
    prometheus_available = False
    print("Prometheus indisponível. Métricas não serão coletadas.")

"""Corrige problemas de certificado SSL"""
ssl._create_default_https_context = ssl._create_unverified_context

"""Configuração do sistema de logging"""
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Importações do PyTorch"""
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("PyTorch indisponível. Modelo LSTM não funcionará.")

"""Importações do MLflow"""
try:
    import mlflow
    import mlflow.pytorch
    mlflow_available = True
except ImportError:
    mlflow_available = False
    print("MLflow indisponível. Rastreamento de experimentos não funcionará.")

"""Criação do diretório static se não existir"""
if not os.path.exists('static'):
    os.makedirs('static')

"""Criação do diretório models se não existir"""
if not os.path.exists('models'):
    os.makedirs('models')

"""Configuração do Flask"""
app = Flask(__name__)

"""Inicialização do Prometheus"""
if prometheus_available:
    metrics = PrometheusMetrics(app)
    metrics.info('appinfos', 'Application info', version='1.0.0')
    MODEL_LOSS = Summary('model_loss_count', 'Loss function during training')
    MODEL_ACCURACY = Summary('model_accuracy_count', 'Accuracy during training')
    MODEL_TRAINING_TIME = Summary('model_training_duration_seconds', 'Time spent training the model')
    MODEL_SIGMOID = Gauge('model_sigmoid_value', 'Sigmoid value computed from the average training loss', ['model'])

def download_with_timeout(symbol, start_date, end_date, timeout=180):
    """
    Faz o download de dados de ações com timeout configurável.
    
    Args:
        symbol (str): Símbolo da ação
        start_date (str): Data de início
        end_date (str): Data de fim
        timeout (int): Timeout em segundos (padrão: 180)
    
    Returns:
        DataFrame ou None: Dados da ação ou None em caso de erro
    """
    try:
        logging.info(f"Fazendo download dos dados para o símbolo {symbol} de {start_date} até {end_date}.")
        
        df = download_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            max_retries=3,
            base_delay=2,
            use_cache=True
        )
        
        if df.empty:
            logging.warning(f"Nenhum dado encontrado para o símbolo {symbol} entre {start_date} e {end_date}.")
        return df
    except Exception as e:
        logging.error(f"Erro ao coletar dados: {e}")
        return None

@app.route('/')
def home():
    """Rota principal que renderiza o formulário."""
    return render_template('form.html')

@app.route('/history')
def history():
    """Página para visualizar predições históricas."""
    symbol = request.args.get('symbol', 'MELI')
    latest_predictions = get_latest_predictions(symbol)
    return render_template('history.html', predictions=latest_predictions, symbol=symbol)

@app.route('/models')
def list_models():
    """Página para visualizar modelos salvos."""
    symbol = request.args.get('symbol', 'MELI')
    models_df = get_saved_models(symbol)
    return render_template('models.html', models=models_df, symbol=symbol)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Rota principal que processa requisições GET e POST.
    
    Handles:
        - Download de dados de ações
        - Treinamento ou carregamento de modelos LSTM
        - Geração de predições
        - Criação de gráficos
    
    Returns:
        str: Template renderizado com resultados
    """
    if request.method == 'POST':
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        symbol = request.form.get('symbol', 'MELI')  
        use_saved_model = request.form.get('use_saved_model') == 'on'
        
        logging.info(f"Recebendo dados para o símbolo {symbol} de {start_date} até {end_date}.")
        logging.info(f"Usar modelo salvo: {use_saved_model}")
        
        df = download_with_timeout(symbol, start_date, end_date)
        if df is None or df.empty:
            logging.error("Erro ao coletar dados. Verifique as datas e tente novamente.")
            return "Erro ao coletar dados. Verifique as datas e tente novamente."
        
        df_close = df[['Close']]
        logging.info("Dados baixados com sucesso.")

        """Salva dados brutos no SQLite"""
        save_raw_data(df_close, symbol)

        """Normaliza os dados"""
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df_close)
        
        """Cria sequências temporais"""
        sequence_length = 60
        X, y = create_sequences(df_scaled, sequence_length)
        
        """Define o número de épocas para treinamento"""
        num_epochs = 55  
        batch_size = 32 
        
        if use_saved_model:
            """Tenta carregar o modelo salvo mais recente"""
            model, saved_scaler, config = load_model(symbol)
            
            if model is not None and saved_scaler is not None:
                logger.info(f"Usando modelo salvo para {symbol}")
                scaler = saved_scaler
                df_scaled = scaler.transform(df_close)
            else:
                logger.warning(f"Nenhum modelo salvo encontrado para {symbol}. Treinando um novo modelo.")
                use_saved_model = False
        
        if not use_saved_model:
            """Cria e treina um novo modelo"""
            model = build_lstm_model(sequence_length)
            
            """Treina o modelo com monitoramento Prometheus"""
            if prometheus_available:
                with MODEL_TRAINING_TIME.time():
                    if mlflow_available:
                        mlflow.set_experiment("LSTM Stock Prediction PyTorch")
                        with mlflow.start_run():
                            mlflow.log_param("sequence_length", sequence_length)
                            mlflow.log_param("num_epochs", num_epochs)
                            mlflow.log_param("batch_size", batch_size)
                            
                            model, loss_history, sigmoid_value = train_lstm(model, X, y, num_epochs, batch_size)
                            
                            mlflow.pytorch.log_model(model, "lstm_model")
                    else:
                        model, loss_history, sigmoid_value = train_lstm(model, X, y, num_epochs, batch_size)
            else:
                if mlflow_available:
                    mlflow.set_experiment("LSTM Stock Prediction PyTorch")
                    with mlflow.start_run():
                        mlflow.log_param("sequence_length", sequence_length)
                        mlflow.log_param("num_epochs", num_epochs)
                        mlflow.log_param("batch_size", batch_size)
                        
                        model, loss_history, sigmoid_value = train_lstm(model, X, y, num_epochs, batch_size)
                        
                        mlflow.pytorch.log_model(model, "lstm_model")
                else:
                    model, loss_history, sigmoid_value = train_lstm(model, X, y, num_epochs, batch_size)

            """Salva métricas do modelo no banco de dados"""
            save_model_metrics(symbol, sequence_length, num_epochs, batch_size, loss_history, sigmoid_value)
            
            """Salva o modelo treinado em arquivo"""
            model_path, scaler_path, config_path = save_model(model, scaler, symbol, sequence_length)
            
            if model_path is not None:
                """Salva informações do modelo no banco de dados"""
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

        """Prediz próximo dia e próximos 5 dias"""
        prediction_next_day, future_predictions_inv = predict_prices(df_scaled, model, scaler, sequence_length, days_ahead=5)

        """Salva predições no banco de dados"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        save_predictions(symbol, current_date, future_predictions_inv)

        logging.info(f"Predição do próximo dia: {prediction_next_day}")
        
        """Calcula mudança percentual para o próximo dia"""
        last_close = df_close['Close'].iloc[-1]
        price_change = prediction_next_day - last_close
        percent_change = (price_change / last_close) * 100
        
        """Cria dicionário com informações de mudança"""
        changes = {
            'last_close': round(last_close, 2),
            'price_change': round(price_change, 2),
            'percent_change': round(percent_change, 2)
        }
        
        """Gráfico 1: Preço de Fechamento ao Longo do Tempo"""
        line_chart_path = 'static/line_chart.png'
        plt.figure(figsize=(12, 6))
        plt.plot(df_close.index, df_close['Close'], label='Preço de Fechamento')
        plt.title('Preço de Fechamento da Ação ao Longo do Tempo')
        plt.xlabel('Data')
        plt.ylabel('Preço de Fechamento')
        plt.legend()
        plt.grid(True)
        plt.savefig(line_chart_path)
        plt.close()

        """Gráfico 2: Histograma da distribuição de preços"""
        hist_chart_path = 'static/hist_chart.png'
        plt.figure(figsize=(10, 6))
        plt.hist(df_close['Close'], bins=50, color='blue', edgecolor='black')
        plt.title('Distribuição do Preço de Fechamento')
        plt.xlabel('Preço de Fechamento')
        plt.ylabel('Frequência')
        plt.grid(True)
        plt.savefig(hist_chart_path)
        plt.close()

        """Gráfico 3: Predição dos Próximos 5 Dias"""
        future_chart_path = 'static/future_chart.png'
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 6), future_predictions_inv, marker='o', linestyle='-', label='Predição')
        plt.title('Predição dos Próximos 5 Dias')
        plt.xlabel('Dia')
        plt.ylabel('Preço Predito')
        plt.grid(True)
        plt.legend()
        plt.savefig(future_chart_path)
        plt.close()

        """Se treinou um novo modelo, mostra a curva de aprendizado"""
        if not use_saved_model:
            """Gráfico 4: Curva de aprendizado (Loss ao longo das épocas)"""
            learning_curve_path = 'static/learning_curve.png'
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, num_epochs+1), loss_history, label='Loss (MSE)')
            plt.title('Curva de Aprendizado (Loss)')
            plt.xlabel('Épocas')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.savefig(learning_curve_path)
            plt.close()

            """Gráfico 5: Métrica Sigmoid"""
            sigmoid_chart_path = 'static/sigmoid_chart.png'
            plt.figure(figsize=(10, 6))
            plt.axhline(y=sigmoid_value, color='r', linestyle='--', label=f'Valor Sigmoid: {sigmoid_value:.4f}')
            plt.title('Valor da Função Sigmoid (baseado na perda média)')
            plt.xlabel('Métrica')
            plt.ylabel('Valor')
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
            """Para modelo salvo, não temos curvas de aprendizado"""
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
    """
    Rota para treinar e salvar um modelo sem fazer predições.
    
    Returns:
        str: Mensagem de sucesso ou erro, ou template de treinamento
    """
    if request.method == 'POST':
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        symbol = request.form.get('symbol', 'MELI')
        num_epochs = int(request.form.get('num_epochs', 55))
        sequence_length = int(request.form.get('sequence_length', 60))
        
        logging.info(f"Treinando modelo para o símbolo {symbol} com dados de {start_date} até {end_date}.")
        
        """Faz download dos dados usando yfinance"""
        df = download_with_timeout(symbol, start_date, end_date)
        if df is None or df.empty:
            logging.error("Erro ao coletar dados. Verifique as datas e tente novamente.")
            return "Erro ao coletar dados. Verifique as datas e tente novamente."
        
        """Seleciona apenas a coluna 'Close'"""
        df_close = df[['Close']]
        logging.info("Dados baixados com sucesso.")
        
        """Salva dados brutos no SQLite"""
        save_raw_data(df_close, symbol)
        
        """Normaliza os dados"""
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df_close)
        
        """Cria sequências temporais"""
        X, y = create_sequences(df_scaled, sequence_length)
        
        """Cria o modelo"""
        model = build_lstm_model(sequence_length)
        
        """Treina o modelo"""
        batch_size = 32
        model, loss_history, sigmoid_value = train_lstm(model, X, y, num_epochs, batch_size)
        
        """Salva métricas do modelo no banco de dados"""
        save_model_metrics(symbol, sequence_length, num_epochs, batch_size, loss_history, sigmoid_value)
        
        """Salva o modelo treinado em arquivo"""
        model_path, scaler_path, config_path = save_model(model, scaler, symbol, sequence_length)
        
        if model_path is not None:
            """Salva informações do modelo no banco de dados"""
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
            
            return f"Modelo treinado e salvo com sucesso em {model_path}"
        else:
            return "Erro ao salvar o modelo treinado."
    
    return render_template('train_model.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=False, host='0.0.0.0')

