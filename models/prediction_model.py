
#Falta dividir em bachs - Sem eficiência //paralelismo
#Early stop
#validação cruzada
#pytorch

#camadas LSTM + Dropout + Dense
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# --------- Criar sequências ---------
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# --------- Modelo LSTM em PyTorch ---------
class LSTMModel(nn.Module):
    def __init__(self, sequence_length):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=150, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=150, hidden_size=150, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(150, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]  # pega só a última saída
        out = self.fc(out)
        return out

# --------- Treinar modelo com EarlyStopping e Cross-Validation ---------
def train_lstm(X, y, sequence_length, n_splits=5, patience=10, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_model = None
    best_val_loss = np.inf

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"Treinando fold {fold+1}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LSTMModel(sequence_length).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1).to(device)

        best_fold_loss = np.inf
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_t)
            loss = criterion(output, y_train_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output = model(X_val_t)
                val_loss = criterion(val_output, y_val_t)

            if val_loss.item() < best_fold_loss:
                best_fold_loss = val_loss.item()
                best_fold_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping no epoch {epoch+1}")
                break

        if best_fold_loss < best_val_loss:
            best_val_loss = best_fold_loss
            best_model = LSTMModel(sequence_length)
            best_model.load_state_dict(best_fold_model)
            best_model = best_model.to(device)

    return best_model

# --------- Prever preços ---------
def predict_prices(df_scaled, model, scaler, sequence_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Previsão do próximo dia
    input_seq = df_scaled[-sequence_length:].reshape(1, sequence_length, 1)
    input_seq_t = torch.tensor(input_seq, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction_scaled = model(input_seq_t).cpu().numpy()
    prediction_next_day = scaler.inverse_transform(prediction_scaled)[0][0]

    # Previsão iterativa para os próximos 5 dias
    future_predictions = []
    input_seq = df_scaled[-sequence_length:].copy()

    for i in range(5):
        input_seq_t = torch.tensor(input_seq.reshape(1, sequence_length, 1), dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(input_seq_t).cpu().numpy()
        future_predictions.append(pred[0][0])

        input_seq = np.append(input_seq[1:], [[pred[0][0]]], axis=0)

    future_predictions_inv = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Calcular variações percentuais
    changes = []
    for i in range(1, len(future_predictions_inv)):
        change = ((future_predictions_inv[i] - future_predictions_inv[i-1]) / future_predictions_inv[i-1]) * 100
        changes.append((i + 1, round(change[0], 2)))

    return prediction_next_day, future_predictions_inv, changes
