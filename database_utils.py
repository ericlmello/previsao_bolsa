"""
Módulo de funções de banco de dados para o sistema de predição de ações.
Este módulo contém todas as funções necessárias para inicializar o banco de dados,
salvar e recuperar dados de preços, predições e métricas de modelos.
"""

import sqlite3
import logging
import pandas as pd
from datetime import datetime
import json

"""Configuração do sistema de logging"""
logger = logging.getLogger(__name__)

def init_db():
    """
    Inicializa o banco de dados SQLite com as tabelas necessárias.
    
    Cria as seguintes tabelas se não existirem:
    - raw_stock_data: dados brutos de preços de ações
    - predictions: predições feitas pelo modelo
    - model_metrics: métricas de performance do modelo
    - saved_models: informações sobre modelos salvos
    """
    with sqlite3.connect('stock_prediction.db') as conn:
        cursor = conn.cursor()
        
        """Tabela para dados brutos de ações"""
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
        
        """Tabela para predições do modelo"""
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

        """Tabela para métricas do modelo"""
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
        
        """Tabela para modelos salvos"""
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
        logger.info("Banco de dados inicializado com sucesso")

def save_raw_data(df, symbol):
    """
    Salva dados brutos de ações no banco de dados SQLite.
    
    Args:
        df (DataFrame): DataFrame contendo os dados de preços
        symbol (str): Símbolo da ação
    """
    try:
        with sqlite3.connect('stock_prediction.db') as conn:
            """Reseta o índice para tornar a data acessível como coluna"""
            df_to_save = df.reset_index()
            
            """Garante que a coluna do índice seja renomeada para 'date'"""
            if 'index' in df_to_save.columns:
                df_to_save.rename(columns={'index': 'date'}, inplace=True)
            elif 'Date' in df_to_save.columns:
                df_to_save.rename(columns={'Date': 'date'}, inplace=True)
            
            """Verifica se a coluna 'date' existe após o renomeamento"""
            if 'date' not in df_to_save.columns:
                df_to_save['date'] = datetime.now().strftime('%Y-%m-%d')
            
            """Garante que 'date' seja convertido para string em formato YYYY-MM-DD"""
            if pd.api.types.is_datetime64_any_dtype(df_to_save['date']):
                df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
            
            """Renomeia 'Close' para 'close_price'"""
            if 'Close' in df_to_save.columns:
                df_to_save.rename(columns={'Close': 'close_price'}, inplace=True)
            
            """Adiciona colunas de símbolo e created_at"""
            df_to_save['symbol'] = symbol
            df_to_save['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            """Seleciona apenas as colunas necessárias"""
            df_to_save = df_to_save[['symbol', 'date', 'close_price', 'created_at']]
            
            """Salva no banco de dados"""
            df_to_save.to_sql('raw_stock_data', conn, if_exists='append', index=False)
            
            logger.info(f"Salvos {len(df_to_save)} registros de dados brutos para {symbol}")
    except Exception as e:
        logger.error(f"Erro ao salvar dados brutos no banco de dados: {e}")

def save_predictions(symbol, prediction_date, future_predictions_inv):
    """
    Salva predições no banco de dados SQLite.
    
    Args:
        symbol (str): Símbolo da ação
        prediction_date (str): Data da predição (formato 'YYYY-MM-DD')
        future_predictions_inv (array): Array de predições para dias futuros
    """
    try:
        with sqlite3.connect('stock_prediction.db') as conn:
            cursor = conn.cursor()
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            """Calcula datas alvo (assumindo que prediction_date é a data atual e predições são para dias futuros)"""
            prediction_dt = datetime.strptime(prediction_date, '%Y-%m-%d')
            
            """Salva cada predição"""
            for i, pred_price in enumerate(future_predictions_inv):
                """Data alvo é i+1 dias a partir da data de predição"""
                target_date = (prediction_dt + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d')
                
                cursor.execute('''
                INSERT INTO predictions (symbol, prediction_date, target_date, predicted_price, created_at)
                VALUES (?, ?, ?, ?, ?)
                ''', (symbol, prediction_date, target_date, float(pred_price), created_at))
            
            conn.commit()
            logger.info(f"Salvadas {len(future_predictions_inv)} predições para {symbol}")
    except Exception as e:
        logger.error(f"Erro ao salvar predições no banco de dados: {e}")

def save_model_metrics(symbol, sequence_length, num_epochs, batch_size, loss_history, sigmoid_value):
    """
    Salva métricas do modelo no banco de dados SQLite.
    
    Args:
        symbol (str): Símbolo da ação
        sequence_length (int): Comprimento da sequência usado no modelo
        num_epochs (int): Número de épocas de treinamento
        batch_size (int): Tamanho do batch usado no treinamento
        loss_history (list): Lista de valores de loss durante o treinamento
        sigmoid_value (float): Valor sigmoid calculado a partir da loss média
    """
    try:
        with sqlite3.connect('stock_prediction.db') as conn:
            cursor = conn.cursor()
            run_date = datetime.now().strftime('%Y-%m-%d')
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            """Converte loss_history para string JSON"""
            loss_history_json = json.dumps(loss_history)
            
            cursor.execute('''
            INSERT INTO model_metrics (symbol, run_date, sequence_length, num_epochs, batch_size, loss_history, sigmoid_value, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, run_date, sequence_length, num_epochs, batch_size, loss_history_json, sigmoid_value, created_at))
            
            conn.commit()
            logger.info(f"Métricas do modelo salvas para {symbol}")
    except Exception as e:
        logger.error(f"Erro ao salvar métricas do modelo no banco de dados: {e}")

def save_model_info_to_db(symbol, timestamp, model_path, scaler_path, config_path, sequence_length, hidden_layer_size, num_layers):
    """
    Salva informações do modelo no banco de dados SQLite.
    
    Args:
        symbol (str): Símbolo da ação
        timestamp (str): Timestamp do modelo salvo
        model_path (str): Caminho para o arquivo do modelo
        scaler_path (str): Caminho para o arquivo do scaler
        config_path (str): Caminho para o arquivo de configuração
        sequence_length (int): Comprimento da sequência do modelo
        hidden_layer_size (int): Tamanho da camada oculta do modelo
        num_layers (int): Número de camadas no modelo
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
            logger.info(f"Informações do modelo salvas no banco de dados para {symbol}")
    except Exception as e:
        logger.error(f"Erro ao salvar informações do modelo no banco de dados: {e}")

def get_latest_predictions(symbol, limit=5):
    """
    Obtém as predições mais recentes do banco de dados.
    
    Args:
        symbol (str): Símbolo da ação
        limit (int): Número de predições a retornar (padrão = 5)
        
    Returns:
        DataFrame: DataFrame com as predições mais recentes
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
        logger.error(f"Erro ao obter predições mais recentes: {e}")
        return pd.DataFrame()

def get_saved_models(symbol):
    """
    Obtém a lista de modelos salvos para um símbolo específico.
    
    Args:
        symbol (str): Símbolo da ação
        
    Returns:
        DataFrame: DataFrame com informações sobre modelos salvos
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
        logger.error(f"Erro ao obter modelos salvos: {e}")
        return pd.DataFrame()
