import pandas as pd
import numpy as np
import os, time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

x_train = pd.read_csv("training_x_data.csv").set_index("Date")
x_train.index = pd.to_datetime(x_train.index)

x_validation = pd.read_csv("validation_x_data.csv").set_index("Date")
x_validation.index = pd.to_datetime(x_validation.index)

y_train = pd.read_csv("training_y_data.csv").set_index("Date")
y_train.index = pd.to_datetime(y_train.index)

y_validation = pd.read_csv("validation_y_data.csv").set_index("Date")
y_validation.index = pd.to_datetime(y_validation.index)

y_cols = y_train.columns

## Initial Training

results = []
models = {}

for y in tqdm(y_cols, desc="Training RF Models", ncols=90):
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train[y])

    models[y] = model

    y_pred = model.predict(x_validation)

    rsq = r2_score(y_validation[y], y_pred)
    mse = mean_squared_error(y_validation[y], y_pred)

    results.append({"y_column": y, "R2": rsq, "MSE": mse})

results = pd.DataFrame(results).sort_values(by="R2", ascending=False).head(70)

y_cols = results['y_column']

## Fine Tuning

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

fine_tuned_results = []
parameters = {}

for y in tqdm(y_cols, desc="Fine Tuning Models", ncols=90):
    best_rsq = -float('inf')
    best_mse = 0
    best_params = None

    for params in ParameterGrid(param_grid):
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train[y])

        y_pred = model.predict(x_validation)

        rsq = r2_score(y_validation[y], y_pred)
        mse = mean_squared_error(y_validation[y], y_pred)

        if rsq > best_rsq:
            best_rsq = rsq
            best_params = params
            best_mse = mse

    fine_tuned_results.append({'y_column': y, 'R2': best_rsq, 'MSE': best_mse})
    parameters[y] = best_params

fine_tuned_results = pd.DataFrame(fine_tuned_results).sort_values(by="R2", ascending=False).head(16)

y_cols = fine_tuned_results['y_column']

## Assessing on profit for validation data

all_profits = {}

for i, y in enumerate(y_cols):
    params = parameters[y]

    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train[y])

    models[y] = model

    size = np.mean(np.abs(y_train[y]))
    y_pred = model.predict(x_validation)

    positions = -5 * y_pred / size

    profit = -1 * y_validation[y] * 100 * positions
    profit.fillna(0)

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

# grid_size = 4

# fig, axes = plt.subplots(nrows = grid_size, ncols = grid_size)
# axes = axes.flatten()

# for i, y in enumerate(y_cols):
#     model = models[y]

#     y_pred = model.predict(x_test)

#     axes[i].plot(y_pred, y_test[y])
#     axes[i].plot(y_test[y], y_test[y])

# fig.subplots_adjust(hspace=1, wspace=0.8)
# plt.show()

all_profits = {}

for i, y in enumerate(y_cols):
    model = models[y]

    size = np.mean(np.abs(y_train[y]))
    y_pred = model.predict(x_test)

    positions = -5 * y_pred / size

    profit = -1 * y_test[y] * 100 * positions
    profit.fillna(0)

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

    dates = x_test.index

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