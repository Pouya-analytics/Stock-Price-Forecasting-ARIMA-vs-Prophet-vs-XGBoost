"""
run_forecast_comparison.py
-----------------------------
Compares three forecasting approaches on real AAPL daily closing price
data (506 trading days, 2015-02-17 to 2017-02-16):

  1. ARIMA (statsmodels)      -- classical statistical time series model
  2. Prophet (Meta/Facebook)  -- structural time series model (trend + seasonality)
  3. XGBoost with lag features -- ML approach, time series reframed as
     tabular supervised learning

All three are evaluated on the SAME held-out test period (the final 20%
of trading days, chronologically -- never random-split, since shuffling
a time series before splitting leaks future information into training).

Metrics: MAE, RMSE, MAPE -- computed once and applied identically to
all three models for a fair comparison.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import load_series, make_lag_features

warnings.filterwarnings("ignore")  # statsmodels/prophet are verbose; suppress for clean output

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# METRICS (computed once, applied identically to all 3 models)
# ---------------------------------------------------------------------
def evaluate(y_true, y_pred, model_name):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"  {model_name:12s} | MAE: {mae:6.3f} | RMSE: {rmse:6.3f} | MAPE: {mape:5.2f}%")
    return {"model": model_name, "mae": mae, "rmse": rmse, "mape": mape}


def main():
    df = load_series()
    n = len(df)
    split_idx = int(n * 0.8)
    train_dates = df.index[:split_idx]
    test_dates = df.index[split_idx:]

    print("=" * 70)
    print("AAPL CLOSING PRICE FORECAST COMPARISON")
    print("=" * 70)
    print(f"Total trading days: {n}")
    print(f"Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
    print(f"Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")
    print()

    results = []
    forecasts = {}  # collected for the comparison plot

    # -------------------------------------------------------------
    # MODEL 1: ARIMA
    # -------------------------------------------------------------
    print("-" * 70)
    print("MODEL 1: ARIMA")
    print("-" * 70)
    from statsmodels.tsa.arima.model import ARIMA
    from pmdarima import auto_arima

    train_series = df["close"].iloc[:split_idx]
    test_series = df["close"].iloc[split_idx:]

    # auto_arima selects (p,d,q) by AIC rather than guessing manually --
    # this is the standard, defensible way to pick ARIMA order rather
    # than eyeballing ACF/PACF plots in a portfolio project.
    auto_model = auto_arima(train_series, seasonal=False, suppress_warnings=True,
                             stepwise=True, trace=False)
    order = auto_model.order
    print(f"  auto_arima selected order: {order}")

    arima_model = ARIMA(train_series, order=order).fit()
    arima_forecast = arima_model.forecast(steps=len(test_series))
    forecasts["ARIMA"] = arima_forecast.values
    results.append(evaluate(test_series.values, arima_forecast.values, "ARIMA"))

    # -------------------------------------------------------------
    # MODEL 2: Prophet
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("MODEL 2: Prophet")
    print("-" * 70)
    from prophet import Prophet

    prophet_train = pd.DataFrame({
        "ds": train_dates,
        "y": df["close"].iloc[:split_idx].values,
    })
    prophet_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
    prophet_model.fit(prophet_train)

    future = pd.DataFrame({"ds": test_dates})
    prophet_pred = prophet_model.predict(future)
    forecasts["Prophet"] = prophet_pred["yhat"].values
    results.append(evaluate(test_series.values, prophet_pred["yhat"].values, "Prophet"))

    # -------------------------------------------------------------
    # MODEL 3: XGBoost with lag features
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("MODEL 3: XGBoost (lag features)")
    print("-" * 70)
    from xgboost import XGBRegressor

    featured = make_lag_features(df)
    # re-split the FEATURED dataframe by date, not by row count, since
    # make_lag_features drops the first ~20 rows (NaN from rolling/lag) --
    # splitting by date keeps train/test boundaries consistent with the
    # other two models even though row counts now differ slightly.
    feat_train = featured[featured.index <= train_dates[-1]]
    feat_test = featured[featured.index > train_dates[-1]]

    feature_cols = [c for c in featured.columns if c != "close"]
    X_train, y_train = feat_train[feature_cols], feat_train["close"]
    X_test, y_test = feat_test[feature_cols], feat_test["close"]

    xgb_model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    forecasts["XGBoost"] = xgb_pred
    # NOTE: feat_test may have slightly fewer rows than test_series due
    # to the date-based resplit above; evaluate against the matching
    # subset, not the full test_series.
    results.append(evaluate(y_test.values, xgb_pred, "XGBoost"))

    # -------------------------------------------------------------
    # Naive baseline -- "tomorrow = today" -- ALWAYS include this.
    # If a sophisticated model can't beat the naive baseline, that's
    # the single most important finding in the whole project, and
    # skipping this comparison is how portfolio projects accidentally
    # overstate a model's value.
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("BASELINE: Naive (random walk: predict tomorrow = today)")
    print("-" * 70)
    naive_pred = df["close"].iloc[split_idx - 1:n - 1].values  # yesterday's actual close
    results.append(evaluate(test_series.values, naive_pred, "Naive"))
    forecasts["Naive"] = naive_pred

    # -------------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    results_df = pd.DataFrame(results).sort_values("rmse")
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_results.csv"), index=False)

    # Save forecasts for plotting
    forecast_df = pd.DataFrame({"date": test_dates[:len(test_series)], "actual": test_series.values})
    for name, preds in forecasts.items():
        # pad/truncate to match test_series length if XGBoost's test set
        # differs slightly in length due to the date-based resplit
        preds_aligned = list(preds) + [np.nan] * (len(test_series) - len(preds))
        forecast_df[name] = preds_aligned[:len(test_series)]
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, "forecasts.csv"), index=False)

    print(f"\nResults saved to {OUTPUT_DIR}/model_comparison_results.csv")
    print(f"Forecasts saved to {OUTPUT_DIR}/forecasts.csv")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    best = results_df.iloc[0]
    naive_row = results_df[results_df["model"] == "Naive"].iloc[0]
    print(f"""
Best model by RMSE: {best['model']} (RMSE={best['rmse']:.3f}, MAPE={best['mape']:.2f}%)
Naive baseline:     RMSE={naive_row['rmse']:.3f}, MAPE={naive_row['mape']:.2f}%

{'WARNING: the naive baseline was NOT beaten by a meaningful margin.' if best['model'] == 'Naive' or best['rmse'] > naive_row['rmse'] * 0.95 else 'The best model meaningfully beat the naive baseline.'}

This is the central, honest finding of this project: daily stock closing
prices are close to a random walk over short horizons, which is exactly
what efficient-market theory predicts. A sophisticated model "winning"
by a tiny margin over "tomorrow = today" is not strong evidence of real
predictive skill -- it could easily be noise. The value of this
comparison isn't proving a model that beats the market; it's demonstrating
the discipline to check against a naive baseline before claims a model
"works," which is a check most portfolio forecasting projects skip.
""")


if __name__ == "__main__":
    main()
