from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load GDP
def load_gdp(csv_path: Path) -> pd.DataFrame:
    # Load GDP data and calculate quarterly growth rate
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["QoQ_Growth_%"] = df["GDP"].pct_change() * 100
    return df[["Date", "QoQ_Growth_%"]]

# 2. Load Yield-Curve & compute 10y–2y spread
from Scripts.dataset_reader import get_yield_curve_rates

def load_yield_spread() -> pd.DataFrame:
    # Load yield curve rates and calculate 10y–2y spread
    yc = get_yield_curve_rates().copy()
    yc["Date"] = pd.to_datetime(yc["Date"])
    yc = yc.sort_values("Date")
    # Compute yield spread
    yc["Spread_10y_2y"] = yc["10 Yr"] - yc["2 Yr"]
    # Convert to quarterly average
    yc["Quarter"] = yc["Date"].dt.to_period("Q")
    spread_q = yc.groupby("Quarter")["Spread_10y_2y"].mean().reset_index()
    spread_q["Date"] = spread_q["Quarter"].dt.to_timestamp()
    spread_q = spread_q[["Date", "Spread_10y_2y"]]
    return spread_q

# 3. Styling
def set_style():
    # Apply consistent Seaborn and Matplotlib styling
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 11

# 4. Visualisation
def make_combined_chart(gdp: pd.DataFrame, spread: pd.DataFrame, out_dir: Path):
    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge GDP and yield spread data by quarter
    merged = pd.merge(gdp, spread, on="Date", how="inner")

    set_style()
    fig, ax1 = plt.subplots()

    # Left y-axis: GDP growth bars
    bar_width = 80
    colors = merged["QoQ_Growth_%"].apply(lambda x: "seagreen" if x >= 0 else "firebrick")
    ax1.bar(merged["Date"], merged["QoQ_Growth_%"], width=bar_width,
            color=colors, alpha=0.7, label="GDP QoQ Growth (%)")
    ax1.set_ylabel("GDP QoQ Growth (%)", color="seagreen")
    ax1.axhline(0, color="black", linewidth=1)

    # Right y-axis: 10y–2y yield spread line
    ax2 = ax1.twinx()
    ax2.plot(merged["Date"],
             merged["Spread_10y_2y"], 
             color="royalblue",
             linewidth=2, 
             label="10-2Y Spread (%)")
    ax2.set_ylabel("10-Year – 2-Year Spread (%)", color="royalblue")
    # Add dashed blue line at zero spread
    ax2.axhline(0, color="royalblue", linestyle="--", linewidth=1.5, alpha=0.7)

    # Format x-axis to show years
    ax1.set_xlabel("Date")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)

    # Add title and legends
    plt.title("GDP Growth vs Yield-Curve Spread (2013–2024)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.tight_layout()
    # Save chart as PNG
    fig.savefig(out_dir / "gdp_vs_yield_spread.png", dpi=300)
    plt.close(fig)
    print(f"Combined chart saved in {out_dir}")

# 5. Main
def main():
    # Define file paths and directories
    base = Path(__file__).parent
    gdp_csv = base / "Datasets" / "GDP_quarterly.csv"
    out_dir = base / "outputs" / "figures"

    # Load data and create combined chart
    gdp = load_gdp(gdp_csv)
    spread = load_yield_spread()
    make_combined_chart(gdp, spread, out_dir)

# Run main
if __name__ == "__main__":
    main()