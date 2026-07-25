Stock Price Forecasting — ARIMA vs Prophet vs XGBoost

I built this to compare three forecasting approaches on the same data with the same evaluation criteria. Most forecasting projects report a MAPE number with nothing to compare it against. That number means nothing without a baseline. So I included one — and it won.

What I compared
ARIMA — classical statistical time series, order selected automatically via AIC (auto_arima), not guessed manually
Prophet — Meta's structural time series model
XGBoost — ML approach, time series reframed as tabular supervised learning with lag features and rolling statistics
Naive baseline — predict tomorrow = today

All four evaluated on the same held-out test period with the same metrics: MAE, RMSE, MAPE. Strict chronological split — no shuffling, which would leak future data into training.

The finding
Model	RMSE	MAPE
Naive	1.245	0.69%
XGBoost	2.869	1.74%
Prophet	5.008	3.21%
ARIMA	7.401	4.21%

None of them beat "tomorrow = today." This is the correct result for daily stock prices over a short horizon — not a failure of implementation. I verified it three ways: auto_arima selected order (0,1,0) — a pure random walk — on its own. XGBoost's dominant feature was lag_1 at 69% importance, meaning it independently rediscovered the naive rule. Disabling Prophet's seasonality changed nothing. All three point to the same conclusion.

Dataset

Real AAPL daily closing prices from the plotly/datasets repository (MIT licensed, 506 trading days, Feb 2015 – Feb 2017). No synthetic data, no disclosure needed.

How to run it
bash
pip install -r requirements.txt
python scripts/run_forecast_comparison.py
python scripts/make_chart.py
Stack

statsmodels · pmdarima · Prophet · XGBoost · pandas · matplotlib
