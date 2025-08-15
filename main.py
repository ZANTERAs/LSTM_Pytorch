# 1. Instalar paquetes si hace falta
# (Descomenta si lo necesitás en tu entorno)
# !pip install yfinance ta scikit-learn torch matplotlib

# 2. Importar librerías
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay

# Opcional: reproducibilidad y device
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Descargar datos de EXC sin group_by
df = yf.download("EXC", period="15y")[['Close', 'Volume']]
df.dropna(inplace=True)

# Soporte robusto: MultiIndex vs columnas simples
def _get_col(frame, colname, ticker="EXC"):
    if isinstance(frame.columns, pd.MultiIndex):
        return frame[(colname, ticker)]
    else:
        return frame[colname]

# 4. Agregar indicadores técnicos
close_prices = _get_col(df, 'Close', 'EXC')
volume_prices = _get_col(df, 'Volume', 'EXC')

df['SMA_20'] = ta.trend.SMAIndicator(close=close_prices, window=20).sma_indicator()
df['EMA_20'] = ta.trend.EMAIndicator(close=close_prices, window=20).ema_indicator()
df['RSI_14'] = ta.momentum.RSIIndicator(close=close_prices, window=14).rsi()
# En tu código usás macd_diff; mantenemos lo mismo:
df['MACD'] = ta.trend.MACD(close=close_prices).macd_diff()
df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=close_prices, volume=volume_prices).on_balance_volume()

df.dropna(inplace=True)

# 5. Normalizar
# NOTA: igual que en tu script, normalizamos TODAS las columnas juntas.
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)  # shape: (n_samples, n_features)

# 6. Crear secuencias para LSTM
X, y = [], []
window = 60
n_features = scaled_data.shape[1]
for i in range(window, len(scaled_data)):
    X.append(scaled_data[i-window:i])   # (window, n_features)
    y.append(scaled_data[i, 0])         # columna 0 = Close normalizado

X = np.array(X, dtype=np.float32)       # (n_samples-window, window, n_features)
y = np.array(y, dtype=np.float32)       # (n_samples-window, )

# 7. Train/Test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convertir a tensores
X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32, device=device).unsqueeze(1)

# 8. Modelo LSTM (equivalente a dos LSTM de 64 unidades)
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)           # out: (batch, seq_len, hidden_dim)
        out_last = out[:, -1, :]        # último paso temporal
        pred = self.fc(out_last)        # (batch, 1)
        return pred

model = LSTMRegressor(input_dim=n_features, hidden_dim=64, num_layers=2).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Entrenamiento (épocas y batch_size como tu Keras: 60 y 32)
EPOCHS = 60
BATCH_SIZE = 32

def batch_iter(X, y, batch_size):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    for i in range(0, len(X), batch_size):
        b = idx[i:i+batch_size]
        yield X[b], y[b]

model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for xb, yb in batch_iter(X_train_t, y_train_t, BATCH_SIZE):
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(X_train_t)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/{EPOCHS} - Loss: {epoch_loss:.6f}")

# 9. Predicciones y desnormalización
model.eval()
with torch.no_grad():
    y_pred_t = model(X_test_t).squeeze(1)  # shape: (n_test,)
y_pred = y_pred_t.detach().cpu().numpy()
y_true = y_test_t.squeeze(1).detach().cpu().numpy()

# Para desnormalizar SOLO la columna Close usando el mismo scaler:
# construimos matrices “dummy” con el mismo nº de features
def inverse_close_from_scaled(scaled_close_1d, scaler, n_features):
    tmp = np.zeros((scaled_close_1d.shape[0], n_features), dtype=np.float64)
    tmp[:, 0] = scaled_close_1d
    inv = scaler.inverse_transform(tmp)[:, 0]
    return inv

y_test_inv = inverse_close_from_scaled(y_true, scaler, n_features)
y_pred_inv = inverse_close_from_scaled(y_pred, scaler, n_features)

# 10. Métricas
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae  = mean_absolute_error(y_test_inv, y_pred_inv)
mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)

print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ MAE:  {mae:.4f}")
print(f"✅ MAPE: {mape*100:.2f}%")

# (Opcional) Plot rápido de test vs pred
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,5))
# plt.plot(df.index[-len(y_test_inv):], y_test_inv, label='Real')
# plt.plot(df.index[-len(y_pred_inv):], y_pred_inv, label='Predicción')
# plt.legend(); plt.title("EXC - Test vs Predicción (PyTorch)"); plt.show()

# 11. Predicción futura (30 días)
last_seq = scaled_data[-window:, :]                     # (window, n_features)
current_seq = last_seq.copy()
future_preds_scaled = []

model.eval()
with torch.no_grad():
    for _ in range(30):
        inp = torch.tensor(current_seq, dtype=torch.float32, device=device).unsqueeze(0)  # (1, window, n_features)
        next_scaled = model(inp).item()   # pred (escala MinMax, solo Close)
        future_preds_scaled.append(next_scaled)

        # Creamos el próximo "row" copiando el último y reemplazando Close por la predicción
        next_row = current_seq[-1].copy()
        next_row[0] = next_scaled
        current_seq = np.vstack([current_seq[1:], next_row])

# Desnormalizar futuras predicciones de Close
future_preds = inverse_close_from_scaled(np.array(future_preds_scaled), scaler, n_features)

# (Opcional) Plot de proyección
# future_idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')  # días hábiles
# plt.figure(figsize=(12,5))
# plt.plot(df.index, _get_col(df, 'Close'), label='Histórico')
# plt.plot(future_idx, future_preds, label='Proyección 30 días')
# plt.legend(); plt.title("EXC - Proyección 30 días (PyTorch)"); plt.show()

# Por si querés ver el array final:
# print(future_preds)

# === Bloque Plotly para PyTorch ===
# 8. Invertir normalización (ya la hicimos antes para test/pred)
# En nuestro flujo, 'future_preds' YA está desnormalizado (en USD).
# Si lo preferís explícito:
future_preds_inv = future_preds.copy()

# Fechas del test: empiezan en el índice 'window'
# y luego aplicamos el split (train/test)
test_start_idx = window + split
test_dates = df.index[test_start_idx:]

# 9/10. y_test_inv y y_pred_inv ya están calculados arriba
# y_test_inv: array 1D en USD
# y_pred_inv: array 1D en USD

# 12. Banda de incertidumbre ±1σ (error en el set de test)
resid = y_test_inv - y_pred_inv
std_error = np.std(resid)
future_upper = future_preds_inv + std_error
future_lower = future_preds_inv - std_error

# Fechas futuras en días hábiles
future_dates = pd.date_range(df.index[-1] + BDay(), periods=30, freq=BDay())

# 14. Gráfico combinado
fig = go.Figure()

# Test real
fig.add_trace(go.Scatter(
    x=test_dates,
    y=y_test_inv.flatten(),
    name='Precio real'
))

# Test predicho
fig.add_trace(go.Scatter(
    x=test_dates,
    y=y_pred_inv.flatten(),
    name='Precio predicho',
    line=dict(dash='dash')
))

# Predicción futura (30 días)
fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_preds_inv,
    mode='lines',
    name="Predicción futura"
))

# Banda de incertidumbre ±1σ
fig.add_trace(go.Scatter(
    x=np.concatenate([future_dates, future_dates[::-1]]),
    y=np.concatenate([future_upper, future_lower[::-1]]),
    fill='toself',
    fillcolor='rgba(255,165,0,0.2)',
    line=dict(color='rgba(255,165,0,0)'),
    hoverinfo="skip",
    name="Banda ±1σ"
))

fig.update_layout(
    title="Predicción de Precio de EXC con LSTM (PyTorch) – Test + 30 días futuros",
    xaxis_title="Fecha",
    yaxis_title="Precio (USD)",
    template="plotly_dark",
    hovermode="x unified",
    height=600
)

fig.show()
