import pandas as pd 
import datetime as dt
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

x_train = pd.read_csv("training_x_data.csv")
x_train["Date"] = pd.to_datetime(x_train["Date"])
x_train.set_index("Date", inplace=True)
x_validation = pd.read_csv("validation_x_data.csv")
x_validation["Date"] = pd.to_datetime(x_validation["Date"])
x_validation.set_index("Date", inplace=True)
x_test = pd.read_csv("test_x_data.csv")
x_test["Date"] = pd.to_datetime(x_test["Date"])
x_test.set_index("Date", inplace=True)

y_train = pd.read_csv("training_y_data.csv")
y_train["Date"] = pd.to_datetime(y_train["Date"])
y_train.set_index("Date", inplace = True)
y_validation = pd.read_csv("validation_y_data.csv")
y_validation["Date"] = pd.to_datetime(y_validation["Date"])
y_validation.set_index("Date", inplace=True)
y_test = pd.read_csv("test_y_data.csv")
y_test["Date"] = pd.to_datetime(y_test["Date"])
y_test.set_index("Date", inplace=True)

y_cols = list(y_train.columns[11:])

y_train_bool = pd.DataFrame()
y_validation_bool = pd.DataFrame()
y_test_bool = pd.DataFrame()

new_y_cols = []

for c in y_cols:
    if c == "Date":
        continue


    # ↓ Below is code for sorting into -1, 0, 1 rather than just 0 and 1
    # thres = 0.05

    # c1_range = max(y_train[c]) - min(y_train[c])
    # c1_med = np.median(y_train[c])

    # c1 = (y_train[c] > (c1_med + c1_range * thres)) * 1 + (y_train[c] < (c1_med - c1_range * thres))*-1

    # c2_range = max(y_validation[c]) - min(y_validation[c])
    # c2_med = np.median(y_validation[c])

    # c2 = (y_validation[c] > (c2_med + c2_range * thres)) * 1 + (y_validation[c] < (c2_med - c2_range * thres))*-1

    c1 = (y_train[c] > 0) * 1
    c2 = (y_validation[c] > 0) * 1
    c3 = (y_test[c] > 0) * 1

    if not np.prod(c1) == 1:
        y_train_bool = y_train_bool.assign(**{c: c1})
        y_validation_bool = y_validation_bool.assign(**{c: c2})
        y_test_bool = y_test_bool.assign(**{c: c3})

        new_y_cols.append(c)

y_cols = new_y_cols

# ↓ Sanity checking the results
# for c in y_cols:
#     t = [1, 0, -1]

#     for ti in t:
#         print(f"{str(round(sum(y_validation_bool[c] == ti)/len(y_validation_bool[c]), 2)).center(6)}", end=" | ")
    
#     print()

# print(max([sum(y_validation_bool[c]) for c in y_cols]), min([sum(y_validation_bool[c]) for c in y_cols]))

# ↓ And stopping the rest of the script from running if we don't care about ML just yet
# raise

results = []

for c in y_cols:
    try:
        model = LogisticRegression(max_iter=200, solver='liblinear')
        model.fit(x_train, y_train_bool[c])

        y_pred = model.predict(x_validation)

        if (not np.prod(y_pred) == 1) and (not np.sum(y_pred) == 0):
            acc = accuracy_score(y_validation_bool[c], y_pred)

            results.append({'y_column': c, 'accuracy': acc})
    except:
        print(np.prod(y_train_bool[c]))

results_df = pd.DataFrame(results)

sorted_results = results_df.sort_values(by='accuracy', ascending=False)

top16 = sorted_results.head(70)

y_cols = list(top16['y_column'])

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

y_cols = list(fine_tuned_results['y_column'])

all_profits = {}
fig, axes = plt.subplots(nrows = 4, ncols = 4)
axes = axes.flatten()
for i, y in enumerate(y_cols):
    y_pred = []
    profit = []

    c = fine_tuned_results["best_c"][i]

    y_train_bool_col = y_train_bool[y]
    y_test_col = y_test[y]
    y_test_bool_col = y_test_bool[y]

    model = LogisticRegression(max_iter=300, solver='liblinear', C=c)
    model.fit(x_train, y_train_bool_col)

    for j in x_test.index:
        x = x_test.loc[[j], :]
        predicted = model.predict(x)

        y_pred.append(predicted[0])

        if predicted > 0:
            position = -5 # Short position 5k
        else: 
            position = 5  # Long position 5k / bp risk

        current_profit = -y_test_col[j]*100*position

        if not np.isnan(current_profit):
            profit.append(current_profit)
        else:
            profit.append(0)
    all_profits[y] = profit

all_profits = pd.DataFrame.from_dict(all_profits)
all_profits = all_profits[all_profits.sum().sort_values(ascending=False).index]

y_cols = all_profits.columns

for i, y in enumerate(y_cols):
    profit = all_profits[y]

    dates = x_test.index
    axes[i].plot(dates, profit, marker='.', alpha=0.7)
    axes[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
    cum_profit = np.sum(profit)
    axes[i].set_title(f'{y} (cumulative profit = ${cum_profit:,.0f})')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Profit ($)')
    axes[i].tick_params(axis='x', rotation=25)


# import textwrap

# values = list(all_profits.sum())
# labels = all_profits.columns

# # Make a DataFrame so we can sort easily
# df = pd.DataFrame({'label': labels, 'value': values}).sort_values('value', ascending=False)

# # Wrap long y-axis labels
# df['wrapped_label'] = ['\n'.join(textwrap.wrap(label, 20)) for label in df['label']]

# plt.figure(figsize=(10, 6))
# bars = plt.barh(df['wrapped_label'], df['value'], color='#4a90e2')

# # Invert y-axis so highest value is at the top
# plt.gca().invert_yaxis()

# # Add dollar labels inside bars
# for bar in bars:
#     width = bar.get_width()
#     plt.text(
#         width * 0.98 if width > 0 else width + (0.02 * df['value'].max()),
#         bar.get_y() + bar.get_height() / 2,
#         f'${width:,.0f}',
#         ha='right' if width > 0 else 'left',
#         va='center',
#         color='white' if abs(width) > df['value'].max() * 0.1 else 'black',
#         fontsize=9,
#         fontweight='bold'
#     )

# plt.xlabel('Profit ($)')
# plt.ylabel('Category')
# plt.title('Total Profit by Category')
# plt.tight_layout()
# plt.show()

# George made this ↓
# plt.figure(figsize=(10,6))
# plt.barh(fine_tuned_results["y_column"], fine_tuned_results["accuracy"])
# plt.xlabel("Accuracy (%)")
# plt.ylabel("Target Variable")
# plt.title("Top 16 Models by Accuracy")
# plt.gca().invert_yaxis()

# # Add numeric labels to the bars
# for i, (v, c) in enumerate(zip(fine_tuned_results["accuracy"], fine_tuned_results["best_c"])):
#     plt.text(v + 0.005, i, f"{v*100:.1f}% (C={c:.3g})", va='center')

# plt.show()

# import textwrap
# import matplotlib.patheffects as path_effects

# # Wrap long y labels
# wrapped_labels = [
#     "\n".join(textwrap.wrap(label, width=20))
#     for label in fine_tuned_results["y_column"]
# ]

# # Normalize accuracies for color mapping
# # norm = plt.Normalize(fine_tuned_results["accuracy"].min()-0.07, fine_tuned_results["accuracy"].max()+0.1)
# # colors = plt.cm.hot(norm(fine_tuned_results["accuracy"]))

# norm = plt.Normalize(fine_tuned_results["accuracy"].min()-0.05, fine_tuned_results["accuracy"].max())
# colors = plt.cm.winter_r(norm(fine_tuned_results["accuracy"]))

# # Create figure and axis explicitly
# fig, ax = plt.subplots(figsize=(10, 6))

# # Create the bar chart
# bars = ax.barh(wrapped_labels, fine_tuned_results["accuracy"]*100, color=colors)
# ax.set_xlabel("Accuracy (%)")
# ax.set_ylabel("Target Variable")
# ax.set_title("Logistic Regression — Best Models")
# ax.invert_yaxis()

# # Add numeric labels inside the bars
# for bar, v, c in zip(bars, fine_tuned_results["accuracy"], fine_tuned_results["best_c"]):
#     txt = ax.text(
#         bar.get_width() - 1,
#         bar.get_y() + bar.get_height() / 2,
#         f"{v*100:.1f}% (C={c:.3g})",
#         va='center',
#         ha='right',
#         color='white',
#         fontsize=11
#     )
#     txt.set_path_effects([
#         path_effects.Stroke(linewidth=2, foreground='black'),
#         path_effects.Normal()
#     ])

fig.subplots_adjust(hspace=0.8, wspace=0.5)
plt.show()