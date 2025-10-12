from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load House Price dataset
def load_house_prices(csv_path: Path) -> pd.DataFrame:
    # Read CSV and prepare quarterly house price data
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # Calculate quarter-over-quarter percentage change
    df["QoQ_Change_%"] = df["ASPUS"].pct_change() * 100
    # Extract year for grouping
    df["Year"] = df["Date"].dt.year
    return df

# 2. Summaries
def summary_tables(df: pd.DataFrame):
    # Compute descriptive statistics and annual averages
    desc = df["ASPUS"].describe()
    annual = df.groupby("Year")["ASPUS"].mean()
    return desc, annual

# 3. Styling
def set_style():
    # Apply Seaborn theme and consistent Matplotlib parameters
    sns.set_theme(style="whitegrid", palette="crest", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 11

# 4. Visualisations
def make_charts(df: pd.DataFrame, out_dir: Path):
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: Average House Price over time
    set_style()
    fig, ax = plt.subplots()
    # Line plot showing house price trend
    sns.lineplot(ax=ax, x="Date", y="ASPUS", data=df,
                 linewidth=2.5, color="royalblue", label="Average House Price")
    # Fill area under curve
    ax.fill_between(df["Date"], df["ASPUS"], alpha=0.15, color="royalblue")
    ax.set_title("US Average House Price (2013–2025)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Sale Price (USD)")
    # Format x-axis for years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)
    ax.legend()
    plt.tight_layout()
    # Save line chart
    fig.savefig(out_dir / "average_house_price_over_time.png", dpi=300)
    plt.close(fig)

    # Chart 2: Quarter-over-Quarter Change (%)
    set_style()
    fig, ax = plt.subplots()
    # Bar chart showing quarterly price changes
    bar_width = 80
    colors = df["QoQ_Change_%"].apply(lambda x: "seagreen" if x >= 0 else "firebrick")
    ax.bar(df["Date"], df["QoQ_Change_%"], width=bar_width, color=colors, alpha=0.7)
    # Add baseline at zero
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Quarter-over-Quarter House Price Change (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Change (%)")
    # Format x-axis for years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    # Save bar chart
    fig.savefig(out_dir / "average_house_price_qoq_change.png", dpi=300)
    plt.close(fig)

# 5. Main
def main():
    # Define paths for datasets and outputs
    base = Path(__file__).parent
    csv_path = base / "Datasets" / "average_house_price_quarterly.csv"
    out_dir = base / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data and generate summary statistics
    df = load_house_prices(csv_path)
    desc, annual = summary_tables(df)

    # Save summary tables to outputs folder
    (base / "outputs").mkdir(parents=True, exist_ok=True)
    desc.to_csv(base / "outputs" / "average_house_price_descriptive_stats.csv")
    annual.to_csv(base / "outputs" / "average_house_price_annual_mean.csv")

    # Create visualisations
    make_charts(df, out_dir)
    print("Charts saved in:", out_dir)

# Run script
if __name__ == "__main__":
    main()

