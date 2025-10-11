import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Scripts.dataset_reader import get_labor_productivity
df = get_labor_productivity()

# Select rows for labor productivity only 
filtered_df = df.filter(like='Labor productivity')

# Set date column as index
filtered_df = filtered_df.set_index(df.iloc[:, 0])

# Simplify sector names
cleaned_sector_names = [col.replace('sector Labor productivity', '') for col in filtered_df.columns]
filtered_df.columns = cleaned_sector_names

plt.figure(figsize=(10, 7.5))
labor_heatmap = sns.heatmap(filtered_df, 
            cmap='RdYlGn', 
            center=0,
            cbar_kws={'label': '% Change in Productivity'},
            linewidths=0.5, 
            linecolor='white')

plt.title('Labor Productivity by Sector Over Time')
plt.ylabel('Year (Quarterly)')
plt.xlabel('Sectors')
plt.xticks(rotation=20)
plt.tight_layout()

for x in range(1, len(filtered_df.columns)):
    plt.axvline(x, color='white', lw=1.2)

plt.show()

