import pandas as pd
import numpy as np
import os, time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

# Read in the data
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
    # For each column we train a model
    # 200 estimators for accuracy, random state 42 for reproducability, and n_jobs -1 so that it runs before the universe ends
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train[y])

    y_pred = model.predict(x_validation)

    # Measure its success
    rsq = r2_score(y_validation[y], y_pred)
    mse = mean_squared_error(y_validation[y], y_pred)

    # Record in dataframe
    results.append({"y_column": y, "R2": rsq, "MSE": mse})

# Sort by R^2 and take the top 70 performers
results = pd.DataFrame(results).sort_values(by="R2", ascending=False).head(70)

# We only care about that ones that made it through, so redefine y_cols
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
    # For each column, we tune the given parameters
    best_rsq = -float('inf')
    best_mse = 0
    best_params = None

    # Try every parameter combination
    for params in ParameterGrid(param_grid):
        # Use the same random state and CPU cores
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train[y])

        y_pred = model.predict(x_validation)

        # Measure new success
        rsq = r2_score(y_validation[y], y_pred)
        mse = mean_squared_error(y_validation[y], y_pred)

        # Optimise for R^2
        if rsq > best_rsq:
            best_rsq = rsq
            best_params = params
            best_mse = mse

    fine_tuned_results.append({'y_column': y, 'R2': best_rsq, 'MSE': best_mse})
    parameters[y] = best_params

# Take the top 16
fine_tuned_results = pd.DataFrame(fine_tuned_results).sort_values(by="R2", ascending=False).head(16)
fine_tuned_results.index = range(len(fine_tuned_results.index))

y_cols = fine_tuned_results['y_column']

## Assessing on profit for validation data

all_profits = {}

# If the model makes sensible predictions on validation data, we allow it through to the test phase
for i, y in enumerate(y_cols):
    params = parameters[y]

    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train[y])

    # Store the model so we don't have to train them again
    models[y] = model

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

# UNUSED BUT FUNCTIONING: Predicted vs Actual graphs
# grid_size = 4

# fig, axes = plt.subplots(nrows = grid_size, ncols = grid_size)
# axes = axes.flatten()

# for i, y in enumerate(y_cols):
#     model = models[y]

#     y_pred = model.predict(x_test)

#     axes[i].plot(y_pred, y_test[y])
    
#     # Line of equality
#     axes[i].plot(y_test[y], y_test[y])

# fig.subplots_adjust(hspace=1, wspace=0.8)
# plt.show()

all_profits = {}

for i, y in enumerate(y_cols):
    # Recall the best model from fine tuning phase
    model = models[y]

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

# creating a table
rf_results = []

for i, y in enumerate(y_cols):
    model = models[y]
    
    # Predictions
    y_pred = model.predict(x_test)
    y_true = y_test[y]
    
    # Align non-NA test values
    mask = ~y_true.isna()
    y_true = y_true[mask]
    y_pred = pd.Series(y_pred, index=y_test.index)[mask]
    
    # Metrics
    test_r2 = r2_score(y_true, y_pred)
    test_mse = mean_squared_error(y_true, y_pred)
    
    # Trading P&L
    size = np.mean(np.abs(y_train[y]))
    positions = -5 * y_pred / size
    profit = -1 * y_true * 100 * positions
    profit = profit.fillna(0)
    cumulative_profit = profit.sum()
    
    # Hit rate (direction accuracy)
    hit_rate = np.mean((y_pred > 0) == (y_true > 0))
    
    rf_results.append({
        "Spread": y,
        "Val_R2": fine_tuned_results.loc[fine_tuned_results["y_column"] == y, "R2"].values[0],
        "Test_R2": test_r2,
        "Val_MSE": fine_tuned_results.loc[fine_tuned_results["y_column"] == y, "MSE"].values[0],
        "Test_MSE": test_mse,
        "Hit_Rate": hit_rate,
        "Total_Profit_$": cumulative_profit
    })

rf_results = pd.DataFrame(rf_results)
print("\nFinal Random Forest Table:\n")
print(rf_results.to_string(index=False))

# Save to CSV for LaTeX
rf_results.to_csv("rf_results_table.csv", index=False)

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