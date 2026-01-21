
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats

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


sectors = ['Nonfarm business sector Labor productivity',
       'Business sector Labor productivity',
       'Nonfinancial corporate sector Labor productivity',
       'Manufacturing sector Labor productivity',
       'Durable manufacturing sector Labor productivity',
       'Nondurable manufacturing sector Labor productivity']

bond_percent = ['1 Mo_pct_change', '3 Mo_pct_change', '6 Mo_pct_change',
                 '1 Yr_pct_change', '2 Yr_pct_change', '3 Yr_pct_change',
                 '5 Yr_pct_change', '7 Yr_pct_change', '10 Yr_pct_change',
                 '20 Yr_pct_change', '30 Yr_pct_change']

# Empty dictionary to store results
corr_results = {}

# Loop through and calculate correlation
for sector in sectors:
    corr_results[sector] = {}
    for bond in bond_percent:
        corr = df_merged[sector].corr(df_merged[bond])
        corr_results[sector][bond] = corr

# Convert to a data frame for display
corr_df = pd.DataFrame(corr_results)

# Clean sector names
corr_df.rename(columns=lambda x: x.replace("sector Labor productivity", ""), inplace=True)

# Clean bond names (index)
corr_df.index = [b.replace("_pct_change", "") for b in corr_df.index]

plt.figure(figsize=(10, 7))

# Plot heatmap
corr_heatmap = sns.heatmap(corr_df, annot=True, cmap='coolwarm_r', center=0, cbar_kws={'label': 'Correlation coefficient (r)'})

# Add axis labels
plt.xlabel("Labor Sectors (% change from previous quarter)")
plt.ylabel("Bond Tenor (% change from previous quarter)")

# Rotate x-axis labels for readability
plt.xticks(rotation=20, ha='right')
plt.yticks(rotation=0)

plt.title("Quarterly Correlation: % Change in Labor Productivity vs Bond Yield Rates", fontsize=14)
plt.tight_layout()

plt.show()
