Cyclone-Preheater-Anomaly-Detection-System

```
Cyclone-Preheater-Anomaly-Detection-System
│
├── feature_engineering.py        # Feature engineering utilities for Streamlit app
├── Algo8.ipynb                   # Full experiment / training notebook
├── app.py                        # Streamlit web application
├── data.csv                      # Training dataset
│
└── models/                       # Saved ML/DL models (auto-generated)
    ├── iso_model.pkl             # Isolation Forest model
    ├── iso_scaler.pkl            # Scaler used for Isolation Forest
    ├── lof_model.pkl             # Local Outlier Factor model (removed due to large size)
    ├── lof_scaler.pkl            # Scaler for LOF model
    ├── lstm_model.h5             # Trained LSTM model
    ├── lstm_scaler.pkl           # Scaler for LSTM input features
    ├── lstm_config.pkl           # LSTM model configuration
    └── lstm_threshold.pkl        # Threshold used for anomaly detection
```
