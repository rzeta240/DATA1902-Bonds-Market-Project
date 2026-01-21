import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

from Scripts.dataset_reader import get_cpi

df = get_cpi()

q75, med, q25 = np.percentile(df["CPIAUCSL_PC1"], [75, 50, 25]) # Obtain quartile values for the dataset, for calculating the IQR and outlier range
iqr = q75 - q25 # Calculate the IQR

outliers = df[ (df["CPIAUCSL_PC1"] > med + 1.5*iqr) | (df["CPIAUCSL_PC1"] < med - 1.5*iqr) ] # Obtain the values that are considered outliers, ie x > ||IQR||*1.5

print(q25, q75) # The 25th and 75th quartile, ie the IQR
print(q75 - q25) # ||IQR||
print(med) # Median
print(np.mean(df["CPIAUCSL_PC1"])) # Mean
print(med - 1.5*iqr, med + 1.5 * iqr) # The outlier range
print(outliers) # A table of the outliers