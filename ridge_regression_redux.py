import pandas as pd 
import datetime as dt
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

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

y_cols = y_train.columns

## Initial training

alp = float(10)

results = []

for y in y_cols:
    model = Ridge(alp)
    model.fit(x_train, y_train[y])

    y_pred = model.predict(x_validation)
    
    mse = mean_squared_error(y_validation[y], y_pred)
    rsq = r2_score(y_validation[y], y_pred)

    results.append({'y_column': y, 'MSE': mse, 'R_squared': rsq})

results = pd.DataFrame(results).sort_values(by="R_squared", ascending=False).head(70)

y_cols = results['y_column']

## Fine Tuning

fine_tuned_results = []

for y in y_cols:
    best_rsq = -float('inf')
    best_alpha = 0

    for alpha in np.logspace(-4, 5, 100):
        model = Ridge(alpha)
        model.fit(x_train, y_train[y])

        y_pred = model.predict(x_validation)

        mse = mean_squared_error(y_validation[y], y_pred)
        rsq = r2_score(y_validation[y], y_pred)

        if rsq > best_rsq:
            best_rsq = rsq
            best_alpha = alpha
    
    fine_tuned_results.append({'y_column': y, 'Best_Alpha': best_alpha, 'R_squared': best_rsq})

fine_tuned_results = pd.DataFrame(fine_tuned_results).sort_values(by='R_squared', ascending=False).head(16)
fine_tuned_results.index = range(len(fine_tuned_results.index))

y_cols = fine_tuned_results['y_column']

## Assessing on profit for validation data

all_profits = {}

for i, y in enumerate(y_cols):
    model = Ridge(fine_tuned_results.loc[i, 'Best_Alpha'])
    model.fit(x_train, y_train[y])

    size = np.mean(np.abs(y_train[y]))
    y_pred = model.predict(x_validation)

    positions = -5 * y_pred / size

    profit = -1 * y_validation[y] * 100 * positions
    profit.fillna(0)

    directional_acc = np.mean(np.sign(y_pred) == np.sign(y_validation[y]))
    mse = mean_squared_error(y_validation[y], y_pred)
    rsq = r2_score(y_validation[y], y_pred)

    print(f"Validation | {y}: MSE={mse:.3f}, R²={rsq:.3f}, Directional Accuracy={directional_acc:.3f}, Profit={profit.sum():.2f}")

    if not (sum(profit) < 0 and np.mean(profit > 0) < 0.6):
        all_profits[y] = profit

all_profits = pd.DataFrame.from_dict(all_profits)
all_profits = all_profits[all_profits.sum().sort_values(ascending=False).index]

y_cols = all_profits.columns

## Running on test data

# Load data --

x_test = pd.read_csv("test_x_data.csv")
x_test["Date"] = pd.to_datetime(x_test["Date"])
x_test.set_index("Date", inplace=True)

y_test = pd.read_csv("test_y_data.csv")
y_test["Date"] = pd.to_datetime(y_test["Date"])
y_test.set_index("Date", inplace=True)

# ------------

all_profits = {}

for i, y in enumerate(y_cols):
    model = Ridge(fine_tuned_results.loc[i, 'Best_Alpha'])
    model.fit(x_train, y_train[y])

    size = np.mean(np.abs(y_train[y]))
    y_pred = model.predict(x_test)

    # Mask NaNs in the test target
    mask = ~y_test[y].isna()
    y_true_clean = y_test[y][mask]
    y_pred_clean = y_pred[mask]

    positions = -5 * y_pred_clean / size
    profit = -1 * y_true_clean * 100 * positions
    profit.fillna(0, inplace=True)

    # Directional accuracy (binary: up/down)
    dir_acc = np.mean((y_pred_clean > 0) == (y_true_clean > 0))

    print(f"Test | {y}: MSE={mean_squared_error(y_true_clean, y_pred_clean):.3f}, "
          f"R²={r2_score(y_true_clean, y_pred_clean):.3f}, "
          f"Directional Accuracy={dir_acc:.3f}, Profit={profit.sum():.2f}")
   
    all_profits[y] = profit

all_profits = pd.DataFrame.from_dict(all_profits)
all_profits = all_profits[all_profits.sum().sort_values(ascending=False).index]

y_cols = all_profits.columns

## Plotting results

print(all_profits.sum())

grid_size = 4

fig, axes = plt.subplots(nrows = grid_size, ncols = grid_size)
axes = axes.flatten()

for i, y in enumerate(y_cols):
    profit = all_profits[y]

    dates = profit.index  # align dates to profit after NaNs are removed

    bars = axes[i].bar(dates, profit, width=25)
    axes[i].set_ylabel("Profit per month ($)")

    ax2 = axes[i].twinx()
    lines = ax2.plot(dates, np.cumsum(profit), marker='.', alpha=0.7, color='red')
    ax2.set_ylabel("Cumulative profit ($)")

    axes[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axes[i].set_title(f'{y}')
    axes[i].tick_params(axis='x', rotation=45)

    ax1_ylims = axes[i].get_ylim()           
    ax1_yratio = ax1_ylims[0] / ax1_ylims[1]  

    ax2_ylims = ax2.get_ylim()           
    ax2_yratio = ax2_ylims[0] / ax2_ylims[1]

    pad = 1.15

    if ax1_yratio < ax2_yratio: 
        ax2.set_ylim(bottom = ax2_ylims[1]*ax1_yratio, top=ax2_ylims[1]*pad)
    else:
        axes[i].set_ylim(bottom = ax1_ylims[1]*ax2_yratio)

fig.subplots_adjust(hspace=1, wspace=0.8)
plt.show()