from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load GDP dataset
def load_gdp(csv_path: Path) -> pd.DataFrame:
    # Read CSV file into DataFrame
    df = pd.read_csv(csv_path)
    # Convert Date column to datetime objects
    df["Date"] = pd.to_datetime(df["Date"])
    # Sort data chronologically
    df = df.sort_values("Date").reset_index(drop=True)
    # Calculate quarter-over-quarter growth in %
    df["QoQ_Growth_%"] = df["GDP"].pct_change() * 100
    # Extract year for grouping
    df["Year"] = df["Date"].dt.year
    return df

# 2. Summaries
def summary_tables(df: pd.DataFrame):
    # Generate descriptive statistics and annual averages
    desc = df["GDP"].describe()
    annual = df.groupby("Year")["GDP"].mean()
    return desc, annual

# 3. Styling
def set_style():
    # Apply Seaborn theme and Matplotlib styling
    sns.set_theme(style="whitegrid", palette="crest", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 11

# 4. Visualisations
def make_charts(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: GDP over time
    set_style()
    fig, ax = plt.subplots()
    # Plot GDP trend line
    sns.lineplot(ax=ax, x="Date", y="GDP", data=df,
                 linewidth=2.5, color="steelblue", label="GDP")
    # Shade area under the curve
    ax.fill_between(df["Date"], df["GDP"], alpha=0.15, color="steelblue")
    ax.set_title("US Real GDP (2013–2024)")
    ax.set_xlabel("Date")
    ax.set_ylabel("GDP (Billions, chained 2017 USD)")
    # Format x-axis to show years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)
    ax.legend()
    plt.tight_layout()
    # Save figure
    fig.savefig(out_dir / "gdp_over_time.png", dpi=300)
    plt.close(fig)

    # Chart 2: Quarter-over-Quarter growth
    set_style()
    fig, ax = plt.subplots()

    # Bar chart showing quarterly GDP growth
    bar_width = 80
    colors = df["QoQ_Growth_%"].apply(lambda x: "seagreen" if x >= 0 else "firebrick")
    ax.bar(df["Date"], df["QoQ_Growth_%"], width=bar_width, color=colors, alpha=0.7)
    # Add horizontal zero line
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Quarter-over-Quarter GDP Growth (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    # Save figure
    fig.savefig(out_dir / "gdp_qoq_growth.png", dpi=300)
    plt.close(fig)

# 5. Main
def main():
    # Define file paths
    base = Path(__file__).parent
    csv_path = base / "Datasets" / "GDP_quarterly.csv"
    out_dir = base / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data and generate summaries
    df = load_gdp(csv_path)
    desc, annual = summary_tables(df)

    # Save summary tables
    (base / "outputs").mkdir(parents=True, exist_ok=True)
    desc.to_csv(base / "outputs" / "gdp_descriptive_stats.csv")
    annual.to_csv(base / "outputs" / "gdp_annual_mean_growth.csv")

    # Create visualisations
    make_charts(df, out_dir)
    print("Charts saved in:", out_dir)

# Run main
if __name__ == "__main__":
    main()
