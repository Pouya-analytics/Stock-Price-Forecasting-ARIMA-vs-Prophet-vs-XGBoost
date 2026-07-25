# Stock-Price-Forecasting-ARIMA-vs-Prophet-vs-XGBoost
ARIMA, Prophet, and XGBoost compared on real AAPL daily prices with a strict chronological split. None beat the naive baseline. Confirmed three ways: auto_arima selected a random walk, XGBoost's top feature was lag_1 at 69%, Prophet seasonality ablation changed nothing.
