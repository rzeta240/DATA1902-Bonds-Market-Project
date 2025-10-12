from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load Unemployment dataset
def load_unemployment(csv_path: Path) -> pd.DataFrame:
    # Load unemployment data and convert monthly to quarterly averages
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Group by quarter and compute average unemployment rate
    df["Quarter"] = df["Date"].dt.to_period("Q")
    quarterly = df.groupby("Quarter")["UNRATE"].mean().reset_index()
    quarterly["Date"] = quarterly["Quarter"].dt.to_timestamp()

    # Calculate quarter-over-quarter percentage change
    quarterly["QoQ_Change_%"] = quarterly["UNRATE"].pct_change() * 100
    # Extract year for annual summaries
    quarterly["Year"] = quarterly["Date"].dt.year
    return quarterly

# 2. Summaries
def summary_tables(df: pd.DataFrame):
    # Generate descriptive statistics and annual averages
    desc = df["UNRATE"].describe()
    annual = df.groupby("Year")["UNRATE"].mean()
    return desc, annual

# 3. Styling
def set_style():
    # Apply Seaborn theme and consistent Matplotlib configuration
    sns.set_theme(style="whitegrid", palette="crest", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 11

# 4. Visualisations
def make_charts(df: pd.DataFrame, out_dir: Path):
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: Average Unemployment Rate over time
    set_style()
    fig, ax = plt.subplots()
    # Line plot of unemployment rate over time
    sns.lineplot(ax=ax, x="Date", y="UNRATE", data=df,
                 linewidth=2.5, color="royalblue", label="Average Unemployment Rate")
    # Fill area under the curve
    ax.fill_between(df["Date"], df["UNRATE"], alpha=0.15, color="royalblue")
    ax.set_title("US Average Unemployment Rate (2013–2025)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate (%)")
    # Format x-axis for years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)
    ax.legend()
    plt.tight_layout()
    # Save line plot
    fig.savefig(out_dir / "average_unemployment_over_time.png", dpi=300)
    plt.close(fig)

    # Chart 2: Quarter-over-Quarter Change (%)
    set_style()
    fig, ax = plt.subplots()
    # Bar chart for quarterly unemployment rate changes
    bar_width = 80
    colors = df["QoQ_Change_%"].apply(lambda x: "firebrick" if x >= 0 else "seagreen")
    ax.bar(df["Date"], df["QoQ_Change_%"], width=bar_width, color=colors, alpha=0.7)
    # Add horizontal line at zero
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Quarter-over-Quarter Unemployment Rate Change (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Change (%)")
    # Format x-axis for years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    # Save bar chart
    fig.savefig(out_dir / "average_unemployment_qoq_change.png", dpi=300)
    plt.close(fig)

# 5. Main
def main():
    # Define file paths for datasets and outputs
    base = Path(__file__).parent
    csv_path = base / "Datasets" / "unemployment_rate_monthly.csv"
    out_dir = base / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset and compute summary statistics
    df = load_unemployment(csv_path)
    desc, annual = summary_tables(df)

    # Save descriptive and annual summary statistics
    (base / "outputs").mkdir(parents=True, exist_ok=True)
    desc.to_csv(base / "outputs" / "average_unemployment_descriptive_stats.csv")
    annual.to_csv(base / "outputs" / "average_unemployment_annual_mean.csv")

    # Create visualisations
    make_charts(df, out_dir)
    print("Charts saved in:", out_dir)

# Run script
if __name__ == "__main__":
    main()

# Summary statistics for Average Unemployment Rate dataset:
# ---------------------------------------------------------
# count : 50.0           → Total number of quarterly observations
# mean  : 5.10           → Average unemployment rate (%)
# std   : 1.71           → Standard deviation (volatility)
# min   : 3.50           → Minimum average unemployment rate recorded
# 25%   : 3.80           → First quartile (25th percentile)
# 50%   : 4.40           → Median (50th percentile)
# 75%   : 5.90           → Third quartile (75th percentile)
# max   : 12.97          → Maximum average unemployment rate recorded


# Yearly Average Unemployment Rates:
# ----------------------------------
# 2013 : 7.3
# 2014 : 6.2
# 2015 : 5.3
# 2016 : 4.9
# 2017 : 4.3
# 2018 : 3.9
# 2019 : 3.7
# 2020 : 9.8
# 2021 : 5.9
# 2022 : 3.6
# 2023 : 3.6
# 2024 : 4.1
# 2025 : 4.2