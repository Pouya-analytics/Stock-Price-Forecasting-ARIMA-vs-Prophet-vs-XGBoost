"""
prepare_data.py
------------------
Loads the real AAPL daily OHLCV dataset (sourced from plotly/datasets,
a real, MIT-licensed, publicly verifiable repository -- NOT synthetic,
unlike most other projects in this portfolio) and builds the lag/rolling
features needed for the XGBoost model.

Source: https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv
Coverage: 2015-02-17 to 2017-02-16, 506 trading days, real AAPL closing
prices (no missing values, verified directly against the source file).
"""
import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "aapl_raw.csv")


def load_series():
    """Returns a clean DataFrame indexed by date with a single 'close'
    column, sorted chronologically (the raw file is already sorted, but
    this is asserted rather than assumed)."""
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "AAPL.Close"]].rename(columns={"AAPL.Close": "close"})
    df = df.sort_values("Date").reset_index(drop=True)
    assert df["Date"].is_monotonic_increasing, "Dates must be sorted for time series work"
    df = df.set_index("Date")
    return df


def make_lag_features(df: pd.DataFrame, lags=(1, 2, 3, 5, 10), windows=(5, 10, 20)) -> pd.DataFrame:
    """
    Builds lag and rolling-window features for the XGBoost model.

    XGBoost has no inherent notion of "time" -- it's a tabular regressor.
    To use it for forecasting, the time series has to be manually
    reframed as a supervised learning problem: predict close[t] from
    close[t-1], close[t-2], ..., plus rolling statistics computed ONLY
    from data strictly before t (no leakage).
    """
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out["close"].shift(lag)
    for window in windows:
        # shift(1) before rolling ensures the window for predicting day t
        # only uses data up to and including day t-1 -- using day t's own
        # close in its own rolling feature would be target leakage.
        out[f"rolling_mean_{window}"] = out["close"].shift(1).rolling(window).mean()
        out[f"rolling_std_{window}"] = out["close"].shift(1).rolling(window).std()

    # day-of-week as a feature -- trading patterns can have weekly effects
    out["day_of_week"] = out.index.dayofweek

    out = out.dropna()
    return out


if __name__ == "__main__":
    df = load_series()
    print(f"Loaded {len(df)} trading days: {df.index.min().date()} to {df.index.max().date()}")
    print(df.head())
    print()
    featured = make_lag_features(df)
    print(f"\nAfter feature engineering: {featured.shape}")
    print(featured.columns.tolist())
    print(featured.head(3))
