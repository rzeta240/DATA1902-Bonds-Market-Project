"""
Creates summary tables and visualisations for US Average House Prices (ASPUS).
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load House Price dataset
def load_house_prices(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["QoQ_Change_%"] = df["ASPUS"].pct_change() * 100
    df["Year"] = df["Date"].dt.year
    return df

# 2. Summaries
def summary_tables(df: pd.DataFrame):
    desc = df["ASPUS"].describe()
    annual = df.groupby("Year")["ASPUS"].mean()
    return desc, annual

# 3. Styling
def set_style():
    sns.set_theme(style="whitegrid", palette="crest", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 11

# 4. Visualisations
def make_charts(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: Average House Price over time
    set_style()
    fig, ax = plt.subplots()
    sns.lineplot(ax=ax, x="Date", y="ASPUS", data=df,
                 linewidth=2.5, color="royalblue", label="Average House Price")
    ax.fill_between(df["Date"], df["ASPUS"], alpha=0.15, color="royalblue")
    ax.set_title("US Average House Price (2013–2025)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Sale Price (USD)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "average_house_price_over_time.png", dpi=300)
    plt.close(fig)

    # Chart 2: Quarter-over-Quarter Change (%)
    set_style()
    fig, ax = plt.subplots()

    bar_width = 80
    colors = df["QoQ_Change_%"].apply(lambda x: "seagreen" if x >= 0 else "firebrick")
    ax.bar(df["Date"], df["QoQ_Change_%"], width=bar_width, color=colors, alpha=0.7)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Quarter-over-Quarter House Price Change (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Change (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(out_dir / "average_house_price_qoq_change.png", dpi=300)
    plt.close(fig)

# 5. Main
def main():
    base = Path(__file__).parent
    csv_path = base / "Datasets" / "average_house_price_quarterly.csv"
    out_dir = base / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_house_prices(csv_path)
    desc, annual = summary_tables(df)

    (base / "outputs").mkdir(parents=True, exist_ok=True)
    desc.to_csv(base / "outputs" / "average_house_price_descriptive_stats.csv")
    annual.to_csv(base / "outputs" / "average_house_price_annual_mean.csv")

    make_charts(df, out_dir)
    print("Charts saved in:", out_dir)

if __name__ == "__main__":
    main()


# Summary statistics for Average House Price dataset:
# ---------------------------------------------------
# count : 50.0           → Total number of quarterly observations
# mean  : 413775.00      → Average house price (USD)
# std   : 66184.72       → Standard deviation (price variability)
# min   : 307400.00      → Minimum average price recorded
# 25%   : 364150.00      → First quartile (25th percentile)
# 50%   : 384300.00      → Median (50th percentile)
# 75%   : 496700.00      → Third quartile (75th percentile)
# max   : 525100.00      → Maximum average price recorded


# Yearly Average House Prices:
# ----------------------------
# 2013 : 321650.00
# 2014 : 345950.00
# 2015 : 350950.00
# 2016 : 359650.00
# 2017 : 381150.00
# 2018 : 382975.00
# 2019 : 379875.00
# 2020 : 387400.00
# 2021 : 452675.00
# 2022 : 516425.00
# 2023 : 507625.00
# 2024 : 507375.00
# 2025 : 513500.00