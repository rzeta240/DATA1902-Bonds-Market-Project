"""
Creates a combined chart: Average House Price change vs 10-year–2-year yield-curve spread.
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Scripts.dataset_reader import get_yield_curve_rates

# 1. Load Average House Price dataset
def load_house_prices(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["QoQ_Change_%"] = df["ASPUS"].pct_change() * 100
    return df[["Date", "QoQ_Change_%"]]

# 2. Load Yield-Curve & compute 10y–2y spread
def load_yield_spread() -> pd.DataFrame:
    yc = get_yield_curve_rates().copy()
    yc["Date"] = pd.to_datetime(yc["Date"])
    yc = yc.sort_values("Date")
    yc["Spread_10y_2y"] = yc["10 Yr"] - yc["2 Yr"]
    yc["Quarter"] = yc["Date"].dt.to_period("Q")
    spread_q = yc.groupby("Quarter")["Spread_10y_2y"].mean().reset_index()
    spread_q["Date"] = spread_q["Quarter"].dt.to_timestamp()
    spread_q = spread_q[["Date", "Spread_10y_2y"]]
    return spread_q

# 3. Styling
def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 11

# 4. Visualisation
def make_combined_chart(house: pd.DataFrame, spread: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge by quarter
    merged = pd.merge(house, spread, on="Date", how="inner")

    set_style()
    fig, ax1 = plt.subplots()

    # Left y-axis: House price change bars
    bar_width = 80  # 80 days wide bars
    colors = merged["QoQ_Change_%"].apply(lambda x: "seagreen" if x >= 0 else "firebrick")
    ax1.bar(merged["Date"], merged["QoQ_Change_%"], width=bar_width,
            color=colors, alpha=0.7, label="House Price QoQ Change (%)")
    ax1.set_ylabel("House Price QoQ Change (%)", color="seagreen")
    ax1.axhline(0, color="black", linewidth=1)

    # Right y-axis: 10y–2y spread line
    ax2 = ax1.twinx()
    ax2.plot(merged["Date"], merged["Spread_10y_2y"], color="royalblue",
             linewidth=2, label="10-2Y Spread (%)")
    ax2.set_ylabel("10-Year – 2-Year Spread (%)", color="royalblue")

    # X-axis formatting
    ax1.set_xlabel("Date")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)

    # Title & legend
    plt.title("Average House Price Change vs Yield-Curve Spread (2013–2025)")

    # Combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_dir / "average_house_price_vs_yield_spread.png", dpi=300)
    plt.close(fig)
    print(f"Combined chart saved in {out_dir}")

# 5. Main
def main():
    base = Path(__file__).parent
    house_csv = base / "Datasets" / "average_house_price_quarterly.csv"
    out_dir = base / "outputs" / "figures"

    house = load_house_prices(house_csv)
    spread = load_yield_spread()
    make_combined_chart(house, spread, out_dir)

if __name__ == "__main__":
    main()