# Predicción de Precios de EXC con LSTM – Versión PyTorch

Este repositorio contiene una **implementación en PyTorch** de un modelo LSTM para la predicción del precio de cierre de la acción **EXC**, incluyendo indicadores técnicos y proyección de 30 días con banda de incertidumbre.

Este código es una **versión reescrita en PyTorch** de un script anterior que había sido desarrollado en **TensorFlow/Keras**, manteniendo la misma lógica y estructura, pero aprovechando la flexibilidad y control que ofrece PyTorch.

---

## Características principales
- Descarga de datos históricos desde **Yahoo Finance** con `yfinance`.
- Cálculo de indicadores técnicos:
  - SMA (Simple Moving Average)
  - EMA (Exponential Moving Average)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - OBV (On-Balance Volume)
- Normalización con **MinMaxScaler**.
- Preparación de secuencias para LSTM.
- Modelo LSTM de 2 capas para regresión.
- Entrenamiento y evaluación con métricas:
  - RMSE
  - MAE
  - MAPE
- Predicción de los próximos **30 días**.
- Banda de incertidumbre ±1σ.
- Visualización con **Plotly**.

---

## Requisitos
```bash
pip install yfinance ta scikit-learn torch plotly
```

---

## Uso
1. Ejecutar el script principal.
2. El modelo entrenará con datos de los últimos 15 años de EXC.
3. Se mostrarán:
   - Gráfico comparando precio real vs predicho (set de test).
   - Proyección de 30 días con banda de incertidumbre.

---

## Salida esperada
- **Métricas** de rendimiento.
- **Gráfico interactivo** en Plotly con:
  - Serie histórica real.
  - Predicción en el test set.
  - Proyección de 30 días futuros.
  - Banda de ±1σ de incertidumbre.

---

# EXC Price Prediction with LSTM – PyTorch Version

This repository contains a **PyTorch implementation** of an LSTM model for predicting the closing price of the **EXC** stock, including technical indicators and a 30-day projection with an uncertainty band.

This code is a **PyTorch rewrite** of a previous script originally built in **TensorFlow/Keras**, keeping the same logic and structure but leveraging PyTorch's flexibility and control.

---

## Main Features
- Download historical data from **Yahoo Finance** using `yfinance`.
- Compute technical indicators:
  - SMA (Simple Moving Average)
  - EMA (Exponential Moving Average)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - OBV (On-Balance Volume)
- Normalize features with **MinMaxScaler**.
- Create sequences for LSTM input.
- Two-layer LSTM regression model.
- Train and evaluate with metrics:
  - RMSE
  - MAE
  - MAPE
- Predict the next **30 days**.
- ±1σ uncertainty band.
- Visualization with **Plotly**.

---

## Requirements
```bash
pip install yfinance ta scikit-learn torch plotly
```

---

## Usage
1. Run the main script.
2. The model will train using the last 15 years of EXC data.
3. It will display:
   - Graph comparing real vs predicted prices (test set).
   - 30-day future projection with an uncertainty band.

---

## Expected Output
- **Performance metrics**.
- **Interactive Plotly chart** showing:
  - Real historical series.
  - Predicted prices on the test set.
  - 30-day future projection.
  - ±1σ uncertainty band.
