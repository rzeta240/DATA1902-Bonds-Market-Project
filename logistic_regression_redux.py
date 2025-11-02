import pandas as pd 
import datetime as dt
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
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

y_cols = list(y_train.columns)

## Making binary data

y_train_bool = pd.DataFrame()
y_validation_bool = pd.DataFrame()
y_test_bool = pd.DataFrame()

new_y_cols = []

for y in y_cols:
    if y == "Date":
        continue

    c1 = (y_train[y] > 0) * 1
    c2 = (y_validation[y] > 0) * 1

    if not np.prod(c1) == 1:
        y_train_bool = y_train_bool.assign(**{y: c1})
        y_validation_bool = y_validation_bool.assign(**{y: c2})

        new_y_cols.append(y)

y_cols = new_y_cols

## Initial training

results = []

for y in y_cols:
    model = LogisticRegression(max_iter=200, solver='liblinear')
    model.fit(x_train, y_train_bool[y])

    y_pred = model.predict(x_validation)

    if (not np.prod(y_pred) == 1) and (not np.sum(y_pred) == 0):
        acc = accuracy_score(y_validation_bool[y], y_pred)

        results.append({'y_column': y, 'accuracy': acc})

results = pd.DataFrame(results).sort_values(by='accuracy', ascending=False).head(70)

y_cols = results['y_column']

## Fine tuning

fine_tuned_results = []

for y in y_cols:
    best_acc = 0
    best_c = 0

    for C in (list(np.logspace(-4, 4, 20)) + [1]):
        model = LogisticRegression(max_iter=300, solver='liblinear', C=C)
        model.fit(x_train, y_train_bool[y])

        y_pred = model.predict(x_validation)

        acc = accuracy_score(y_validation_bool[y], y_pred)

        if acc > best_acc:
            best_acc = acc
            best_c = C
    
    fine_tuned_results.append({'y_column': y, 'accuracy': best_acc, 'best_c': best_c})

fine_tuned_results = pd.DataFrame(fine_tuned_results).sort_values(by='accuracy', ascending=False).head(16)

fine_tuned_results.index = range(len(fine_tuned_results.index))

y_cols = fine_tuned_results['y_column']

## Assessing based on profit

all_profits = {}

for i, y in enumerate(y_cols):
    model = LogisticRegression(max_iter=300, solver='liblinear', C=fine_tuned_results.loc[i, 'best_c'])
    model.fit(x_train, y_train_bool[y])

    y_pred = model.predict(x_validation)

    positions = -5 * (y_pred * 2 - 1) 

    profit = -1 * y_validation[y] * 100 * positions
    profit.fillna(0)

    acc = accuracy_score(y_validation_bool[y], y_pred)
    f1 = f1_score(y_validation_bool[y], y_pred)
    total_profit = profit.sum()
    print(f"Validation | {y}: Accuracy={acc:.3f}, F1={f1:.3f}, Profit={total_profit:.2f}")

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

# Create binary test labels (same as training and validation)
y_test_bool = pd.DataFrame()
for y in y_cols:
    c3 = (y_test[y] > 0) * 1
    y_test_bool = y_test_bool.assign(**{y: c3})


# ------------

all_profits = {}

for i, y in enumerate(y_cols):
    model = LogisticRegression(max_iter=300, solver='liblinear', C=fine_tuned_results.loc[i, 'best_c'])
    model.fit(x_train, y_train_bool[y])

    y_pred = model.predict(x_test)

    positions = -5 * (y_pred * 2 - 1) 

    profit = -1 * y_test[y] * 100 * positions
    profit.fillna(0)

    acc = accuracy_score(y_test_bool[y], y_pred)
    f1 = f1_score(y_test_bool[y], y_pred)
    total_profit = profit.sum()
    print(f"Test | {y}: Accuracy={acc:.3f}, F1={f1:.3f}, Profit={total_profit:.2f}")

    all_profits[y] = profit

all_profits = pd.DataFrame.from_dict(all_profits)
all_profits = all_profits[all_profits.sum().sort_values(ascending=False).index]

y_cols = all_profits.columns

## Plotting results

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