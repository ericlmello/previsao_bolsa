"""
Módulo de funções de banco de dados para o sistema de predição de ações.
Este módulo contém todas as funções necessárias para inicializar o banco de dados,
salvar e recuperar dados de preços, predições e métricas de modelos.
"""

import sqlite3
import pandas as pd
import logging
from datetime import datetime
import json

# Configuração do logging
logger = logging.getLogger(__name__)

def init_db():
    """Inicializa o banco de dados SQLite com as tabelas necessárias."""
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        
        # Tabela para dados brutos de ações
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            close_price REAL NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Tabela para predições
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            day_ahead INTEGER NOT NULL,
            predicted_price REAL NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Tabela para métricas do modelo
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            sequence_length INTEGER NOT NULL,
            num_epochs INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            final_loss REAL NOT NULL,
            sigmoid_value REAL NOT NULL,
            loss_history TEXT NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Tabela para informações dos modelos salvos
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            model_timestamp TEXT NOT NULL,
            model_path TEXT NOT NULL,
            scaler_path TEXT NOT NULL,
            config_path TEXT NOT NULL,
            sequence_length INTEGER NOT NULL,
            hidden_size INTEGER NOT NULL,
            num_layers INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Banco de dados inicializado com sucesso.")
        
    except Exception as e:
        logger.error(f"Erro ao inicializar banco de dados: {e}")

def save_raw_data(df_close, symbol):
    """
    Salva dados brutos de ações no banco de dados.
    
    Args:
        df_close (DataFrame): DataFrame com dados de fechamento
        symbol (str): Símbolo da ação
    """
    try:
        conn = sqlite3.connect('stock_data.db')
        
        # Prepara os dados para inserção
        data_to_insert = []
        for index, row in df_close.iterrows():
            data_to_insert.append((
                symbol,
                index.strftime('%Y-%m-%d'),
                float(row['Close']),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
        
        # Insere os dados
        cursor = conn.cursor()
        cursor.executemany('''
        INSERT INTO raw_stock_data (symbol, date, close_price, timestamp) 
        VALUES (?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        logger.info(f"Dados brutos salvos no banco de dados para {symbol}: {len(data_to_insert)} registros")
        
    except Exception as e:
        logger.error(f"Error saving raw data to database: {e}")

def save_predictions(symbol, current_date, future_predictions):
    """
    Salva predições no banco de dados.
    
    Args:
        symbol (str): Símbolo da ação
        current_date (str): Data atual
        future_predictions (list): Lista com predições para os próximos dias
    """
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        
        # Insere uma predição para cada dia futuro
        for day_ahead, prediction in enumerate(future_predictions, 1):
            cursor.execute('''
            INSERT INTO predictions (symbol, prediction_date, day_ahead, predicted_price) 
            VALUES (?, ?, ?, ?)
            ''', (symbol, current_date, day_ahead, float(prediction)))
        
        conn.commit()
        conn.close()
        logger.info(f"Predições salvas no banco de dados para {symbol}")
        
    except Exception as e:
        logger.error(f"Erro ao salvar predições: {e}")

def save_model_metrics(symbol, sequence_length, num_epochs, batch_size, loss_history, sigmoid_value):
    """
    Salva métricas do modelo no banco de dados.
    
    Args:
        symbol (str): Símbolo da ação
        sequence_length (int): Comprimento da sequência
        num_epochs (int): Número de épocas
        batch_size (int): Tamanho do batch
        loss_history (list): Histórico de loss
        sigmoid_value (float): Valor sigmoid
    """
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        
        final_loss = loss_history[-1] if loss_history else 0.0
        
        cursor.execute('''
        INSERT INTO model_metrics 
        (symbol, sequence_length, num_epochs, batch_size, final_loss, sigmoid_value, loss_history) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, sequence_length, num_epochs, batch_size, final_loss, sigmoid_value, json.dumps(loss_history)))
        
        conn.commit()
        conn.close()
        logger.info(f"Métricas do modelo salvas para {symbol}")
        
    except Exception as e:
        logger.error(f"Erro ao salvar métricas do modelo: {e}")

def save_model_info_to_db(symbol, timestamp, model_path, scaler_path, config_path, 
                         sequence_length, hidden_size, num_layers):
    """
    Salva informações do modelo no banco de dados.
    
    Args:
        symbol (str): Símbolo da ação
        timestamp (str): Timestamp do modelo
        model_path (str): Caminho do arquivo do modelo
        scaler_path (str): Caminho do arquivo do scaler
        config_path (str): Caminho do arquivo de configuração
        sequence_length (int): Comprimento da sequência
        hidden_size (int): Tamanho da camada oculta
        num_layers (int): Número de camadas
    """
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO saved_models 
        (symbol, model_timestamp, model_path, scaler_path, config_path, 
         sequence_length, hidden_size, num_layers) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, timestamp, model_path, scaler_path, config_path, 
              sequence_length, hidden_size, num_layers))
        
        conn.commit()
        conn.close()
        logger.info(f"Informações do modelo salvas para {symbol}")
        
    except Exception as e:
        logger.error(f"Erro ao salvar informações do modelo: {e}")

def get_latest_predictions(symbol, limit=10):
    """
    Recupera as predições mais recentes para um símbolo.
    
    Args:
        symbol (str): Símbolo da ação
        limit (int): Número máximo de registros
    
    Returns:
        list: Lista com as predições
    """
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT prediction_date, day_ahead, predicted_price, timestamp 
        FROM predictions 
        WHERE symbol = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        ''', (symbol, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{'prediction_date': row[0], 'day_ahead': row[1], 
                'predicted_price': row[2], 'timestamp': row[3]} for row in results]
        
    except Exception as e:
        logger.error(f"Erro ao recuperar predições: {e}")
        return []

def get_saved_models(symbol):
    """
    Recupera informações dos modelos salvos para um símbolo.
    
    Args:
        symbol (str): Símbolo da ação
    
    Returns:
        DataFrame: DataFrame com informações dos modelos
    """
    try:
        conn = sqlite3.connect('stock_data.db')
        
        query = '''
        SELECT model_timestamp, model_path, sequence_length, 
               hidden_size, num_layers, created_at 
        FROM saved_models 
        WHERE symbol = ? 
        ORDER BY created_at DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao recuperar modelos salvos: {e}")
        return pd.DataFrame()
