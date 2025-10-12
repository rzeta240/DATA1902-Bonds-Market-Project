from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Scripts.dataset_reader import get_yield_curve_rates

# 1. Load Unemployment dataset
def load_unemployment(csv_path: Path) -> pd.DataFrame:
    # Load unemployment data and convert monthly data to quarterly averages
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Group by quarter and calculate average unemployment
    df["Quarter"] = df["Date"].dt.to_period("Q")
    quarterly = df.groupby("Quarter")["UNRATE"].mean().reset_index()
    quarterly["Date"] = quarterly["Quarter"].dt.to_timestamp()

    # Compute quarter-over-quarter percentage change
    quarterly["QoQ_Change_%"] = quarterly["UNRATE"].pct_change() * 100
    return quarterly[["Date", "UNRATE", "QoQ_Change_%"]]

# 2. Load Yield-Curve data and compute 10y–2y spread
def load_yield_spread() -> pd.DataFrame:
    # Retrieve yield curve data and calculate 10-year minus 2-year spread
    yc = get_yield_curve_rates().copy()
    yc["Date"] = pd.to_datetime(yc["Date"])
    yc = yc.sort_values("Date")
    yc["Spread_10y_2y"] = yc["10 Yr"] - yc["2 Yr"]

    # Convert to quarterly averages
    yc["Quarter"] = yc["Date"].dt.to_period("Q")
    spread_q = yc.groupby("Quarter")["Spread_10y_2y"].mean().reset_index()
    spread_q["Date"] = spread_q["Quarter"].dt.to_timestamp()
    return spread_q[["Date", "Spread_10y_2y"]]

# 3. Styling
def set_style():
    # Apply Seaborn theme and consistent Matplotlib parameters
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 11

# 4. Visualisation
def make_combined_chart(unemp: pd.DataFrame, spread: pd.DataFrame, out_dir: Path):
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge unemployment and yield spread data on date
    merged = pd.merge(unemp, spread, on="Date", how="inner")
    set_style()
    fig, ax1 = plt.subplots()

    # Left y-axis: bar chart for unemployment rate change
    bar_width = 80
    colors = merged["QoQ_Change_%"].apply(lambda x: "firebrick" if x > 0 else "seagreen")
    ax1.bar(merged["Date"], merged["QoQ_Change_%"], width=bar_width,
            color=colors, alpha=0.7, label="Unemployment QoQ Change (%)")

    # Apply symmetric log scale to handle extreme spikes while preserving near-zero detail
    ax1.set_yscale("symlog", linthresh=1)
    ax1.set_ylabel("Quarterly Change in Unemployment Rate (symlog scale, %)", color="firebrick")
    ax1.axhline(0, color="black", linewidth=1)

    # Right y-axis: yield curve spread as a line chart
    ax2 = ax1.twinx()
    ax2.plot(merged["Date"], merged["Spread_10y_2y"], color="royalblue",
             linewidth=2.5, label="10y–2y Spread (%)")
    ax2.set_ylabel("10-Year – 2-Year Spread (%)", color="royalblue")
    # Add dashed horizontal line at zero yield spread
    ax2.axhline(0, color="royalblue", linestyle="--", linewidth=1.5, alpha=0.7)

    # Format x-axis with yearly ticks and rotated labels
    ax1.set_xlabel("Date")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)

    # Add title and combined legend
    plt.title("Quarterly Change in Unemployment Rate vs Yield-Curve Spread (2013–2025)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.tight_layout()
    # Save figure to output directory
    fig.savefig(out_dir / "average_unemployment_vs_yield_spread.png", dpi=300)
    plt.close(fig)
    print(f"Combined chart saved in {out_dir}")

# 5. Main
def main():
    # Define dataset and output paths
    base = Path(__file__).parent
    unemp_csv = base / "Datasets" / "unemployment_rate_monthly.csv"
    out_dir = base / "outputs" / "figures"

    # Load data and create combined chart
    unemp = load_unemployment(unemp_csv)
    spread = load_yield_spread()
    make_combined_chart(unemp, spread, out_dir)

# Run script
if __name__ == "__main__":
    main()