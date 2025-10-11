import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Scripts.dataset_reader import get_labor_productivity
df = get_labor_productivity()

# Select columns with labor productivity
df_productivity = df.loc[:, df.columns.str.endswith('Labor productivity')]

# Display summary statistics for every sector
for col in df_productivity.columns:
    min_value = df_productivity[col].min()
    print(f"Minimum of '{col}': {min_value:.2f} %")
    max_value = df_productivity[col].max()
    print(f"Maximum of '{col}': {max_value:.2f} %")
    mean_value = df_productivity[col].mean()
    print(f"Mean of '{col}': {mean_value:.2f} %")
    median_value = df_productivity[col].median()
    print(f"Median of '{col}': {median_value:.2f} %")
    sd_value = df_productivity[col].std(ddof=0)
    print(f"Standard deviation of '{col}': {sd_value:.2f} %")