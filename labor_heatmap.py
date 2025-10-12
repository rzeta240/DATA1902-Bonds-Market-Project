import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Scripts.dataset_reader import get_labor_productivity
df = get_labor_productivity()

# Select columns for labor productivity only 
filtered_df = df.filter(like='Labor productivity')

# Set date column as index
filtered_df = filtered_df.set_index(df.iloc[:, 0])

for i in range(len(filtered_df.columns)): # Convert to Log Plot 
    logged = filtered_df.iloc[:, i]
    logged = np.log10(np.abs(logged))*np.sign(logged)
    filtered_df.iloc[:, i] = logged

filtered_df = filtered_df.fillna(0)

# Simplify sector names
cleaned_sector_names = [col.replace('sector Labor productivity', '') for col in filtered_df.columns]
filtered_df.columns = cleaned_sector_names

# Convert dates to only years for y axis ticks
years = [2013 + (i // 4) for i in range(len(filtered_df))]

plt.figure(figsize=(10, 7.5))
labor_heatmap = sns.heatmap(filtered_df, 
            cmap='RdYlGn', 
            center=0,
            cbar_kws={'label': '% Change in Productivity (Log10)'},
            linewidths=0.5, 
            linecolor='white', # Add separation between points for readability
            yticklabels=years)

plt.title('Labor Productivity by Sector Over Time (2013-2024)')
plt.ylabel('Year (Quarterly)')
plt.xlabel('Sectors')
plt.xticks(rotation=20)
plt.tight_layout()

# Add only one year tick for every four quarters for readability
plt.yticks([i for i in range(0, len(filtered_df), 4)], labels=[years[i] for i in range(0, len(filtered_df), 4)])


plt.show()
