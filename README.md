# EXC Price Prediction with LSTM – PyTorch Version
*(Español abajo | Spanish below)*

This repository provides a **PyTorch rewrite** of a stock price prediction pipeline you originally built in **TensorFlow/Keras**. It keeps the same logic (data download, technical indicators, scaling, sequence generation, LSTM training, test evaluation, and 30‑day forecasting) while using PyTorch’s `nn.Module` and manual training loop. It also includes two uncertainty options: a simple **±1σ band** from test residuals and a **Monte Carlo (MC) Dropout** approach for probabilistic uncertainty.

---

## 🔧 Requirements
```bash
pip install yfinance ta scikit-learn torch plotly
```

---

## 📦 What’s included
- Data loading from Yahoo Finance (`yfinance`)
- Technical indicators: SMA, EMA, RSI, MACD (diff), OBV
- MinMax scaling
- Sequence generation (`window=60` by default)
- Two-layer LSTM regressor (`hidden=64`)
- Metrics: RMSE, MAE, MAPE
- 30-day rolling forecast
- **Uncertainty**:
  - Deterministic band: ±1σ using test residuals
  - **MC Dropout**: mean & PI via stochastic forward passes

---

## 🚀 Quick start
1. Run the main script (PyTorch version).  
2. You’ll get: metrics on the test set, a Plotly chart (real vs predicted), and a 30‑day forecast.  
3. By default, the chart can include a **±1σ** band. You can also enable **MC Dropout** (see below).

---

## 📈 Uncertainty options

### Option A — Simple ±1σ band (already in the code)
Compute the standard deviation of residuals on the test set and use it as a constant band for the 30‑day forecast:
```python
resid = y_test_inv - y_pred_inv
std_error = np.std(resid)
future_upper = future_preds_inv + std_error
future_lower = future_preds_inv - std_error
```
Pros: trivial to add.  
Cons: assumes homoscedastic noise and ignores future uncertainty compounding.

---

### Option B — Monte Carlo Dropout (probabilistic)
Use dropout **at inference** to sample from an approximate posterior predictive distribution.

**1) Add dropout to the model**
You can add it either inside the LSTM stack (via `dropout=` for inter-layer connections) and/or as an explicit layer before the final `Linear`:

```python
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, p_drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=p_drop if num_layers > 1 else 0.0  # dropout between LSTM layers
        )
        self.dropout = nn.Dropout(p_drop)  # explicit dropout before the head
        self.fc = nn.Linear(hidden_dim, 1)

        # (Optional) Initialize weights for stability
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out, _ = self.lstm(x)
        out_last = out[:, -1, :]
        out_last = self.dropout(out_last)  # keep for MC at inference
        return self.fc(out_last)
```

**2) Enable dropout at inference and sample T times**
```python
def mc_dropout_predict(model, seq_array, T=200):
    """
    seq_array: numpy array of shape (window, n_features) for the *last* window
    Returns arrays (mean, std, samples) in original price units (USD).
    """
    model.train()  # IMPORTANT: keep dropout active
    # ensure no gradient update
    for p in model.parameters():
        p.requires_grad_(False)

    current_seq = seq_array.copy()
    samples = []

    with torch.no_grad():
        for _ in range(T):
            # roll forward 30 steps with stochastic dropout each step
            cur = current_seq.copy()
            preds_scaled = []
            for _ in range(30):
                inp = torch.tensor(cur, dtype=torch.float32, device=next(model.parameters()).device).unsqueeze(0)
                next_scaled = model(inp).item()
                preds_scaled.append(next_scaled)
                next_row = cur[-1].copy()
                next_row[0] = next_scaled
                cur = np.vstack([cur[1:], next_row])

            # inverse-transform Close
            preds_usd = inverse_close_from_scaled(np.array(preds_scaled), scaler, n_features)
            samples.append(preds_usd)

    samples = np.array(samples)  # (T, 30)
    mean = samples.mean(axis=0)
    std  = samples.std(axis=0)
    return mean, std, samples
```

**3) Build prediction intervals**
```python
mc_mean, mc_std, _ = mc_dropout_predict(model, last_seq, T=200)
future_lower = mc_mean - 1.96 * mc_std
future_upper = mc_mean + 1.96 * mc_std
```
> Tip: You can plot `mc_mean` instead of `future_preds_inv`, and the shaded band using `future_lower`/`future_upper`.

**Notes**
- `model.train()` is intentional at inference to **keep dropout active**.
- We still use `torch.no_grad()` so no gradients/memory are tracked.
- Increase `T` (e.g., 300–1000) for smoother intervals (more compute).

---

## 🗂 Structure (suggested)
```
project/
├─ README.md
├─ requirements.txt  (optional)
├─ main_pytorch.py   (training + test eval + 30d forecast + ±1σ)
├─ mc_dropout_utils.py (optional helper for MC)
└─ figs/             (optional: save charts)
```

---

## 🧪 Reproducibility
- Seeds are set for NumPy and PyTorch in the script.
- Results can vary due to stochastic training and MC sampling.

---

## 🇪🇸 Español

Este repositorio es una **reescritura en PyTorch** del pipeline de predicción de precios de acciones que construiste previamente en **TensorFlow/Keras**. Mantiene la misma lógica (descarga de datos, indicadores técnicos, escalado, generación de secuencias, entrenamiento LSTM, evaluación en test y proyección a 30 días) y agrega dos enfoques de **incertidumbre**: una banda **±1σ** a partir de residuales en test, y **Dropout Monte Carlo (MC)** para intervalos probabilísticos.

### Requisitos
```bash
pip install yfinance ta scikit-learn torch plotly
```

### Flujo general
1. Descarga EXC (15 años), calcula SMA/EMA/RSI/MACD/OBV.
2. Escala con MinMax y arma ventanas (por defecto 60 pasos).
3. Entrena un LSTM de dos capas (64 unidades).
4. Evalúa en test (RMSE/MAE/MAPE) y grafica con Plotly.
5. Proyecta 30 días hacia adelante.
6. Incertidumbre:
   - **±1σ**: banda fija basada en residuales del test.
   - **MC Dropout**: intervalos con muestreo estocástico en inferencia.

### MC Dropout (resumen)
1. Agregar `Dropout` al modelo (en LSTM inter-capas y/o antes de la capa final).  
2. En inferencia, mantener `model.train()` para que Dropout siga activo y muestrear **T** trayectorias.  
3. Calcular media y desviación estándar para construir **PI** (por ejemplo, 95% con ±1.96·σ).

### Notas
- `torch.no_grad()` evita el cálculo de gradientes en inferencia.
- A mayor `T`, mejores estimaciones de los intervalos (más costo computacional).
- Los resultados pueden variar por la naturaleza estocástica del entrenamiento y del muestreo MC.

---

**License**: MIT (or adapt to your needs).
