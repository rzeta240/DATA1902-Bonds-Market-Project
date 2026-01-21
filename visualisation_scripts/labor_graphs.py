import pandas as pd
import matplotlib.pyplot as plt

from Scripts.dataset_reader import get_labor_productivity

df = get_labor_productivity()

for i in range( len( df.columns ) - 1 ):
    if i == 0:
        continue
    plt.plot(df["Date"], df.iloc[:, i], label = df.columns[i])

plt.legend(bbox_to_anchor=(1, 1))
plt.subplots_adjust(right=0.4)
plt.title("The Data Visualisation Equivalent Of A Shitpost", wrap = True)
plt.show()