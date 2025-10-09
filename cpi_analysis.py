import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

from Scripts.dataset_reader import get_cpi

df = get_cpi()

x = df["Date"]
y = df.iloc[:, 1]
int_y = np.cumsum(y)

q75, med, q25 = np.percentile(df["CPIAUCSL_PC1"], [75, 50, 25])
iqr = q75 - q25

outliers = df[ (df["CPIAUCSL_PC1"] > med + 1.5*iqr) | (df["CPIAUCSL_PC1"] < med - 1.5*iqr) ]

print(q25, q75)
print(q75 - q25)
print(med)
print(np.mean(df["CPIAUCSL_PC1"]))
print(med - 1.5*iqr, med + 1.5 * iqr)
for i in range(len(outliers["Date"])):
    print(f"\\hline\n{outliers.iloc[i, 0].year} Q{(outliers.iloc[i, 0].month - 1)/3 + 1:.0f} & {outliers.iloc[i, 1]} \\\\")