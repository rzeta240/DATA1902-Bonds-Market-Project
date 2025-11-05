import pandas as pd
import plotly.graph_objects as go

from Scripts.dataset_reader import get_yield_curve_rates, get_labor_productivity

df_yield = get_yield_curve_rates()
df_labor = get_labor_productivity()

# Convert to timetime type and aggregate to quarterly 
df_yield['Date'] = pd.to_datetime(df_yield['Date'])
df_yield['year_quarter'] = df_yield['Date'].dt.to_period('Q')
yield_quarterly = df_yield.groupby('year_quarter', as_index=False).mean(numeric_only=True)

# Convert to datetime type and create year_quarter merge key
df_labor['Date'] = pd.to_datetime(df_labor['Date'])
df_labor['year_quarter'] = df_labor['Date'].dt.to_period('Q')

# Create merged dataframe
df_merged = pd.merge(
    df_labor, 
    yield_quarterly, 
    on='year_quarter', 
    how='inner'  # keep only quarters present in both datasets
)

# Calculate change from previous quarter
bonds = ['1 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']

for bond in bonds:
    df_merged[f'{bond}_pct_change'] = df_merged[bond].pct_change() * 100

df_merged = df_merged.dropna().reset_index(drop=True)

bond_percent = ['1 Mo_pct_change', '3 Mo_pct_change', '6 Mo_pct_change',
                 '1 Yr_pct_change', '2 Yr_pct_change', '3 Yr_pct_change',
                 '5 Yr_pct_change', '7 Yr_pct_change', '10 Yr_pct_change',
                 '20 Yr_pct_change', '30 Yr_pct_change']

all_metrics = sorted({
    col.split("sector ")[1]
    for col in df_labor.columns
    if "sector " in col
})

# Select heatmaps with at least 3 sectors
metrics = [m for m in all_metrics if sum(f"sector {m}" in c for c in df_labor.columns) >= 3]

# Empty dictionary to store results
corr_data = {}

# Loop through and calculate correlation
for metric in metrics:
    # Get all columns matching the metric (e.g. unit labor costs)
    sectors = [col for col in df_labor.columns if col.endswith(metric)]
    if not sectors:
        continue

    corr_results = {}
    for sector in sectors:
        corr_results[sector] = {}
        for bond in bond_percent:
            corr = df_merged[sector].corr(df_merged[bond])
            corr_results[sector][bond] = corr

    # Convert to DataFrame for this metric
    corr_df = pd.DataFrame(corr_results)
    corr_df.rename(columns=lambda x: x.replace(f"sector {metric}", ""), inplace=True)
    corr_df.index = [b.replace("_pct_change", "") for b in corr_df.index]
    corr_data[metric] = corr_df

fig = go.Figure()

for i, metric in enumerate(metrics):
    fig.add_trace(go.Heatmap(
        z=corr_data[metric].values,
        x=corr_data[metric].columns,
        y=corr_data[metric].index,
        colorscale='RdBu',
        zmin=-0.6,
        zmax=0.6,
        zmid=0,
        colorbar=dict(title='Correlation (r)'),
        name=metric,
        hovertemplate="Sector: %{x}<br>Bond: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        xgap=1,
        ygap=1,
        visible=(i == 0)
    ))

# Slider steps
steps = [
    dict(
        method="update",
        args=[
            {"visible": [j == i for j in range(len(metrics))]}],
            label=f"{metric}"
    )
    for i, metric in enumerate(metrics)
]


# Layout
fig.update_layout(
    title=dict(
        text="Quarterly Correlation: % Change in Labor Sectors vs Bond Yield Rates",
        x=0.5,
        xanchor='center'
    ),
    # margin=dict(t=220), in case we want to make slider at top
    xaxis_title="Labor Sectors (% change from previous quarter)",
    yaxis_title="Bond Tenor (% change from previous quarter)",
    sliders=[dict(
        active=0,
        currentvalue={"prefix": "Metric: ", 
                      "font": {"size": 16, "family": "Arial, sans-serif", "color": "black"} 
        },
        pad={"t": 50},
        y=-0.3,     
        steps=steps,
        font={"color": "rgba(0,0,0,0)"}  # make static labels bottom of slider invisible
    )],
    width=1000,
    height=680,
    plot_bgcolor='white',
    paper_bgcolor='white',
)

fig.show()
