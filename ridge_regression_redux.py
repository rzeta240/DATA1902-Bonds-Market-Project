import pandas as pd 
import datetime as dt
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

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

y_cols = y_train.columns

## Initial training

# Common value of alpha to test on all y columns
alp = float(10)

results = []

for y in y_cols:
    # For each column we train a model
    model = Ridge(alp)
    model.fit(x_train, y_train[y])

    y_pred = model.predict(x_validation)
    
    # Measure its success
    mse = mean_squared_error(y_validation[y], y_pred)
    rsq = r2_score(y_validation[y], y_pred)

    # Record in dataframe
    results.append({'y_column': y, 'MSE': mse, 'R_squared': rsq})

# Sort by R^2 and take the top 70 performers
results = pd.DataFrame(results).sort_values(by="R_squared", ascending=False).head(70)

# We only care about that ones that made it through, so redefine y_cols
y_cols = results['y_column']

## Fine Tuning

fine_tuned_results = []

for y in y_cols:
    # For each column, we tune the value of alpha
    best_rsq = -float('inf')
    best_alpha = 0

    # Try each value
    for alpha in np.logspace(-4, 5, 100):
        model = Ridge(alpha)
        model.fit(x_train, y_train[y])

        y_pred = model.predict(x_validation)

        mse = mean_squared_error(y_validation[y], y_pred)
        rsq = r2_score(y_validation[y], y_pred)

        # Find best R^2
        if rsq > best_rsq:
            best_rsq = rsq
            best_alpha = alpha
    
    fine_tuned_results.append({'y_column': y, 'Best_Alpha': best_alpha, 'R_squared': best_rsq})

# Take the top 16
fine_tuned_results = pd.DataFrame(fine_tuned_results).sort_values(by='R_squared', ascending=False).head(16)
fine_tuned_results.index = range(len(fine_tuned_results.index))

y_cols = fine_tuned_results['y_column']

## Assessing on profit for validation data

all_profits = {}

# If the model makes sensible predictions on validation data, we allow it through to the test phase
for i, y in enumerate(y_cols):
    model = Ridge(fine_tuned_results.loc[i, 'Best_Alpha'])
    model.fit(x_train, y_train[y])

    # Average size of y data, so we can scale for volatility
    size = np.mean(np.abs(y_train[y]))
    y_pred = model.predict(x_validation)

    positions = -5 * y_pred / size

    # Determine profit
    profit = -1 * y_validation[y] * 100 * positions
    profit.fillna(0)

    # If it's not negative and it was right most of the time, we allow it through 
    if not (sum(profit) < 0 and np.mean(profit > 0) < 0.6):
        all_profits[y] = profit

# We test on everything that made it through
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
    # Recall the best alpha value from fine tuning phase
    model = Ridge(fine_tuned_results.loc[i, 'Best_Alpha'])
    model.fit(x_train, y_train[y])

    # Again scale for volatility
    size = np.mean(np.abs(y_train[y]))
    y_pred = model.predict(x_test)

    positions = -5 * y_pred / size

    # Determine profit
    profit = -1 * y_test[y] * 100 * positions
    profit.fillna(0)

    all_profits[y] = profit

# Sort by profit
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

    # Bar chart of profit per month
    bars = axes[i].bar(dates, profit, width=25)
    axes[i].set_ylabel("Profit per month ($)")

    # On a twin axis with a cumulative profit line chart
    ax2 = axes[i].twinx()
    lines = ax2.plot(dates, np.cumsum(profit), marker='.', alpha=0.7, color='red')
    ax2.set_ylabel("Cumulative profit ($)")

    # Line at y = 0
    axes[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axes[i].set_title(f'{y}')
    axes[i].tick_params(axis='x', rotation=45)

    # We scale the axes of each graph so that their zero points are aligned
    ax1_ylims = axes[i].get_ylim()           
    ax1_yratio = ax1_ylims[0] / ax1_ylims[1]  

    ax2_ylims = ax2.get_ylim()           
    ax2_yratio = ax2_ylims[0] / ax2_ylims[1]

    # Add some room at the top too
    pad = 1.15

    if ax1_yratio < ax2_yratio: 
        ax2.set_ylim(bottom = ax2_ylims[1]*ax1_yratio, top=ax2_ylims[1]*pad)
    else:
        axes[i].set_ylim(bottom = ax1_ylims[1]*ax2_yratio)

# Adjust subplot spacing
fig.subplots_adjust(hspace=1, wspace=0.8)
plt.show()