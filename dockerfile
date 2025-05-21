FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar os requisitos e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY app.py database_utils.py model_utils.py ./
COPY templates/ ./templates/
COPY static/ ./static/

# Criar diretórios necessários
RUN mkdir -p models static

# Expor a porta para a aplicação Flask
EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["python", "app.py"]