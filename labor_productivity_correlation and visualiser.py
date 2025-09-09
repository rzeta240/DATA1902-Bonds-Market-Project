# THIS FILE IS NOT PART OF THE SUBMISSION FOR STAGE 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Scripts.dataset_reader import get_labor_productivity

labor = get_labor_productivity()

measure = "Labor compensation"

col = [i for i in range(len(labor.columns)) if measure in labor.columns[i]]

productivity = labor.iloc[:, col].transpose()
productivity.columns = labor["Date"]

print(productivity)

correlation_matrix = np.corrcoef(productivity)

print(correlation_matrix)

for i in col:
    plt.plot(labor.iloc[:, i])

plt.title(measure)

plt.show()