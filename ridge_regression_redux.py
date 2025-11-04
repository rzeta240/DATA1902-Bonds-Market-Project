import pandas as pd 
import datetime as dt
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
def train_ridge_model(look_forward_window, structure):
    # Read in the data
    x_train = pd.read_csv("training_x_data.csv")
    x_train["Date"] = pd.to_datetime(x_train["Date"])
    x_train.set_index("Date", inplace=True)
    x_validation = pd.read_csv("validation_x_data.csv")
    x_validation["Date"] = pd.to_datetime(x_validation["Date"])
    x_validation.set_index("Date", inplace=True)

    y_train = pd.read_csv("training_y_data.csv")
    y_train["Date"] = pd.to_datetime(y_train["Date"])
    y_train.set_index("Date", inplace = True)
    y_validation = pd.read_csv("validation_y_data.csv")
    y_validation["Date"] = pd.to_datetime(y_validation["Date"])
    y_validation.set_index("Date", inplace=True)

    y_col = f"{structure}_{look_forward_window}d_change"


    # For each column, we tune the value of alpha
    best_rsq = -float('inf')
    best_alpha = 0

    # Try each value
    for alpha in np.logspace(0, 5, 100):
        model = Ridge(alpha)
        model.fit(x_train, y_train[y_col])

        y_pred = model.predict(x_validation)

        mse = mean_squared_error(y_validation[y_col], y_pred)
        rsq = r2_score(y_validation[y_col], y_pred)

        # Find best R^2
        if rsq > best_rsq:
            best_rsq = rsq
            best_alpha = alpha


    ## Running on test data

    # Load data --

    x_test = pd.read_csv("test_x_data.csv")
    x_test["Date"] = pd.to_datetime(x_test["Date"])
    x_test.set_index("Date", inplace=True)

    y_test = pd.read_csv("test_y_data.csv")
    y_test["Date"] = pd.to_datetime(y_test["Date"])
    y_test.set_index("Date", inplace=True)

    # ------------

    for i in enumerate(y_col):
        # Recall the best alpha value from fine tuning phase
        model = Ridge(best_alpha)
        model.fit(x_train, y_train[y_col])

        # Again scale for volatility
        size = np.mean(np.abs(y_train[y_col]))
        y_pred = model.predict(x_test)

        positions = -5 * y_pred / size

        # Determine profit
        profit = -1 * y_test[y_col] * 100 * positions
        profit.fillna(0)

        # Mask NaNs in the test target
        mask = ~y_test[y_col].isna()
        y_true_clean = y_test[y_col][mask]
        y_pred_clean = y_pred[mask]

        positions = -5 * y_pred_clean / size
        profit = -1 * y_true_clean * 100 * positions
        profit.fillna(0, inplace=True)

        # Directional accuracy (binary: up/down)
        dir_acc = np.mean((y_pred_clean > 0) == (y_true_clean > 0))

        return profit, mean_squared_error(y_true_clean, y_pred_clean), r2_score(y_true_clean, y_pred_clean), dir_acc, y_true_clean, y_pred_clean


## Plotting results

# print(all_profits.sum())

# grid_size = 4

# fig, axes = plt.subplots(nrows = grid_size, ncols = grid_size)
# axes = axes.flatten()

# for i, y in enumerate(y_cols):
#     profit = all_profits[y]

#     dates = profit.index  # align dates to profit after NaNs are removed

#     # Bar chart of profit per month
#     bars = axes[i].bar(dates, profit, width=25)
#     axes[i].set_ylabel("Profit per month ($)")

#     # On a twin axis with a cumulative profit line chart
#     ax2 = axes[i].twinx()
#     lines = ax2.plot(dates, np.cumsum(profit), marker='.', alpha=0.7, color='red')
#     ax2.set_ylabel("Cumulative profit ($)")

#     # Line at y = 0
#     axes[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
#     axes[i].set_title(f'{y}')
#     axes[i].tick_params(axis='x', rotation=45)

#     # We scale the axes of each graph so that their zero points are aligned
#     ax1_ylims = axes[i].get_ylim()           
#     ax1_yratio = ax1_ylims[0] / ax1_ylims[1]  

#     ax2_ylims = ax2.get_ylim()           
#     ax2_yratio = ax2_ylims[0] / ax2_ylims[1]

#     # Add some room at the top too
#     pad = 1.15

#     if ax1_yratio < ax2_yratio: 
#         ax2.set_ylim(bottom = ax2_ylims[1]*ax1_yratio, top=ax2_ylims[1]*pad)
#     else:
#         axes[i].set_ylim(bottom = ax1_ylims[1]*ax2_yratio)

# # Adjust subplot spacing
# fig.subplots_adjust(hspace=1, wspace=0.8)
# plt.show()