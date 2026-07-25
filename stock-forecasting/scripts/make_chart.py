"""
make_chart.py
----------------
Generates the forecast comparison chart (actual vs. each model's
predictions over the test period) from output/forecasts.csv, which is
produced by run_forecast_comparison.py. Kept as a separate script so the
core analysis doesn't depend on matplotlib being available.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")


def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "forecasts.csv"), parse_dates=["date"])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["date"], df["actual"], label="Actual", color="black", linewidth=2.2, zorder=5)
    ax.plot(df["date"], df["Naive"], label="Naive (yesterday=today)", color="#888888",
             linestyle="--", linewidth=1.5)
    ax.plot(df["date"], df["XGBoost"], label="XGBoost", color="#2E86AB", linewidth=1.5)
    ax.plot(df["date"], df["Prophet"], label="Prophet", color="#E07A5F", linewidth=1.5)
    ax.plot(df["date"], df["ARIMA"], label="ARIMA", color="#81B29A", linewidth=1.5)

    ax.set_title("AAPL Closing Price Forecast Comparison (Test Period)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "forecast_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")

    # Second chart: bar comparison of RMSE/MAPE across models
    results = pd.read_csv(os.path.join(OUTPUT_DIR, "model_comparison_results.csv"))
    results = results.sort_values("rmse")

    fig2, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#888888" if m == "Naive" else "#2E86AB" for m in results["model"]]
    axes[0].bar(results["model"], results["rmse"], color=colors)
    axes[0].set_title("RMSE by Model (lower is better)")
    axes[0].set_ylabel("RMSE (USD)")
    axes[1].bar(results["model"], results["mape"], color=colors)
    axes[1].set_title("MAPE by Model (lower is better)")
    axes[1].set_ylabel("MAPE (%)")
    for ax_ in axes:
        ax_.grid(axis="y", alpha=0.3)
    fig2.tight_layout()

    out_path2 = os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    fig2.savefig(out_path2, dpi=150)
    print(f"Chart saved to {out_path2}")


if __name__ == "__main__":
    main()
