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

print(outliers)