# ZEEL Stock Price Forecasting with LSTM

Predicts all 11 features of ZEEL (Zee Entertainment) stock for the next 5 trading days using a multi-output LSTM built in PyTorch.

---

## What it does

- Takes the last **10 trading days** as input and forecasts the next **5 trading days**
- Predicts all 11 features: `Prev Close`, `Open`, `High`, `Low`, `Last`, `Close`, `VWAP`, `Volume`, `Turnover`, `Trades`, `%Deliverable`
- Uses separate output heads for price-like vs. volume-like features
- Evaluates with MSE, RMSE, MAE, R², MAPE, and Directional Accuracy

---

## How to run

1. Place `ZEEL.csv` in the project folder
2. Install dependencies:
```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn
```
3. Open and run the notebook top to bottom:
```bash
jupyter notebook "Faraz_Zeel_LSTM_(11_Features).ipynb"
```

---

## Project structure

```
├── Faraz_Zeel_LSTM_(11_Features).ipynb   # Main notebook
├── ZEEL.csv                               # Raw stock data (required)
├── lstm_zeel.pth                          # Saved model weights (generated)
└── scaler_zeel.pkl                        # Fitted scaler (generated)
```

---

## Model highlights

- 2-layer LSTM with hidden size 128 and dropout 0.35
- HuberLoss to handle large price spikes in historical data
- AdamW optimizer with CosineAnnealingWarmRestarts scheduler
- Early stopping with patience of 25 epochs
- 70 / 15 / 15 train/val/test split with no data leakage

---

## Requirements

Python 3.8+ · PyTorch · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn
