from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load GDP dataset
def load_gdp(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["QoQ_Growth_%"] = df["GDP"].pct_change() * 100
    df["Year"] = df["Date"].dt.year
    return df

# 2. Summaries
def summary_tables(df: pd.DataFrame):
    desc = df["GDP"].describe()
    annual = df.groupby("Year")["GDP"].mean()
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

    # Chart 1: GDP over time
    set_style()
    fig, ax = plt.subplots()
    sns.lineplot(ax=ax, x="Date", y="GDP", data=df,
                 linewidth=2.5, color="steelblue", label="GDP")
    ax.fill_between(df["Date"], df["GDP"], alpha=0.15, color="steelblue")
    ax.set_title("US Real GDP (2013–2024)")
    ax.set_xlabel("Date")
    ax.set_ylabel("GDP (Billions, chained 2017 USD)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "gdp_over_time.png", dpi=300)
    plt.close(fig)

    #Chart 2: Quarter-over-Quarter growth
    set_style()
    fig, ax = plt.subplots()

    # Use Matplotlib bar with fixed width (80 days)
    bar_width = 80     # adjust this for more/less spacing
    colors = df["QoQ_Growth_%"].apply(lambda x: "seagreen" if x >= 0 else "firebrick")
    ax.bar(df["Date"], df["QoQ_Growth_%"], width=bar_width, color=colors, alpha=0.7)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Quarter-over-Quarter GDP Growth (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(out_dir / "gdp_qoq_growth.png", dpi=300)
    plt.close(fig)

# 5. Main
def main():
    base = Path(__file__).parent
    csv_path = base / "Datasets" / "GDP_quarterly.csv"
    out_dir = base / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_gdp(csv_path)
    desc, annual = summary_tables(df)

    (base / "outputs").mkdir(parents=True, exist_ok=True)
    desc.to_csv(base / "outputs" / "gdp_descriptive_stats.csv")
    annual.to_csv(base / "outputs" / "gdp_annual_mean_growth.csv")

    make_charts(df, out_dir)
    print("Charts saved in:", out_dir)

if __name__ == "__main__":
    main()

# Summary statistics for GDP dataset:
# -----------------------------------
# count : 50.0           → Total number of quarterly observations
# mean  : 22113.93       → Average GDP value
# std   : 4202.23        → Standard deviation (measure of variability)
# min   : 16648.19       → Minimum GDP recorded
# 25%   : 18572.38       → First quartile (25th percentile)
# 50%   : 21014.73       → Median (50th percentile)
# 75%   : 25658.22       → Third quartile (75th percentile)
# max   : 30353.90       → Maximum GDP recorded


# Yearly GDP values:
# ------------------
# 2013 : 16880.68
# 2014 : 17608.14
# 2015 : 18295.02
# 2016 : 18804.91
# 2017 : 19612.10
# 2018 : 20656.52
# 2019 : 21539.98
# 2020 : 21354.10
# 2021 : 23681.17
# 2022 : 26006.89
# 2023 : 27720.71
# 2024 : 29184.89
# 2025 : 30157.97