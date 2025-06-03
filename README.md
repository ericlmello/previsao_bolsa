Sistema de Previsão de Preços de Ações com LSTM
  
Um aplicativo web desenvolvido com Flask para prever preços de ações utilizando modelos LSTM (Long Short-Term Memory) implementados com PyTorch.
📊 Características
    • Download Automático: Obtém dados históricos de ações usando a API yfinance 
    • Modelo LSTM: Implementa deep learning para previsão de séries temporais 
    • Visualizações: Gera gráficos e visualizações dos dados e previsões 
    • Armazenamento: Salva dados históricos e previsões em banco de dados SQLite 
    • Monitoramento: Integração opcional com MLflow e Prometheus para rastreamento de experimentos e métricas 


🚀 Instalação
Pré-requisitos
    • Python 3.6+ (recomendado Python 3.8+) 
    • Pip (gerenciador de pacotes Python) 
Instalação básica
# Clonar o repositório
git clone https://github.com/ericlmello/previsao_bolsa.git
cd stock-prediction-lstm

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
Aceleração com GPU (opcional)
Para treinamento mais rápido com GPU CUDA:
pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
🔧 Uso
Iniciar o servidor
python app.py
Acesse o aplicativo em http://localhost:5000
Fluxo de uso básico
    1. Digite o símbolo da ação (ex: "QQQ", "AAPL") 
    2. Selecione o intervalo de datas para análise 
    3. Clique em "Enviar" para iniciar o processo 
    4. Visualize os resultados da previsão e os gráficos 
🧠 Modelo LSTM
O sistema utiliza redes neurais recorrentes LSTM (Long Short-Term Memory) para capturar padrões em séries temporais financeiras:
    • Arquitetura: LSTM com camada oculta configurável 
    • Parâmetros padrão: 
        ◦ Tamanho da sequência: 60 dias 
        ◦ Épocas de treinamento: 55 
        ◦ Tamanho do lote: 32 
    • Métricas: MSE (Erro Quadrático Médio), RMSE, MAE 
📂 Estrutura do Projeto
stock-prediction-app/
├── app.py                 # Aplicativo principal (Flask + lógica)
├── model_utils.py         # Utilitários para o modelo LSTM
├── requirements.txt       # Dependências do projeto
├── static/                # Arquivos estáticos
├── templates/             # Templates HTML
│   ├── form.html          # Formulário de entrada
│   ├── result.html        # Página de resultados
│   └── history.html       # Histórico de previsões
└── stock_prediction.db    # Banco de dados SQLite
🛠️ Tecnologias Utilizadas
    • Backend: Python, Flask 
    • Machine Learning: PyTorch, NumPy, scikit-learn 
    • Dados: Pandas, yfinance 
    • Banco de Dados: SQLite 
    • Visualização: Matplotlib 
    • Monitoramento: MLflow, Prometheus (opcional) 
⚙️ Configurações Avançadas
Para modificar os parâmetros do modelo, edite as seguintes variáveis em app.py:
# Tamanho da sequência (dias anteriores para análise)
sequence_length = 150

# Número de épocas para treinamento
num_epochs = 55

# Tamanho do lote para treinamento
batch_size = 32
🔍 Monitoramento e Métricas
MLflow
O sistema pode utilizar MLflow para rastrear experimentos. Para visualizar as métricas após a execução:
mlflow ui
Acesse o dashboard MLflow em http://localhost:5000
Prometheus (opcional)
Se configurado, o Prometheus coleta métricas como:
    • Perdas durante o treinamento 
    • Tempos de treinamento 
    • Valores de função sigmóide 
⚠️ Limitações
    • O modelo foca apenas no preço de fechamento, não considerando fatores externos 
    • Previsões mais precisas para curto prazo (próximos dias) 
    • Performance de treinamento limitada em hardware sem GPU 
