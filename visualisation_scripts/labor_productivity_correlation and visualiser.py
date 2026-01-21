# THIS FILE IS NOT PART OF THE SUBMISSION FOR STAGE 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Scripts.dataset_reader import get_labor_productivity

labor = get_labor_productivity().iloc[:, 1:]

plt.figure(figsize=(10, 8))  # increase figure size
sns.heatmap(labor.corr(), fmt=".2f", cmap="coolwarm")

plt.tight_layout()  # adjust margins so labels fit
plt.show()