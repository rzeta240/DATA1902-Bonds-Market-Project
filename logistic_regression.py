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

y_train = pd.read_csv("training_y_data.csv")
y_train["Date"] = pd.to_datetime(y_train["Date"])
y_train.set_index("Date", inplace = True)
y_validation = pd.read_csv("validation_y_data.csv")
y_validation["Date"] = pd.to_datetime(y_validation["Date"])
y_validation.set_index("Date", inplace=True)

y_cols = list(y_train.columns[11:])

y_train_bool = pd.DataFrame()
y_validation_bool = pd.DataFrame()

new_y_cols = []

for c in y_cols:
    if c == "Date":
        continue

    c1 = (y_train[c] > 0)*1
    c2 = (y_validation[c] > 0)*1

    if not np.prod(c1) == 1:
        y_train_bool = y_train_bool.assign(**{c: c1})
        y_validation_bool = y_validation_bool.assign(**{c: c2})

        new_y_cols.append(c)

y_cols = new_y_cols

results = []

for c in y_cols:
    try:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=200, solver="liblinear")
        )
        model.fit(x_train, y_train_bool[c])

        y_pred = model.predict(x_validation)
        acc = accuracy_score(y_validation_bool[c], y_pred)

        results.append({'y_column': c, 'accuracy': acc})
    except:
        print(np.prod(y_train_bool[c]))

results_df = pd.DataFrame(results)

sorted_results = results_df.sort_values(by='accuracy', ascending=False)

top16 = sorted_results.head(16)

# George made this ↓
plt.figure(figsize=(10,6))
plt.barh(top16["y_column"], top16["accuracy"])
plt.xlabel("Accuracy (%)")
plt.ylabel("Target Variable")
plt.title("Top 16 Models by Accuracy")
plt.gca().invert_yaxis()

# Add numeric labels to the bars
for i, v in enumerate(top16["accuracy"]):
    plt.text(v + 0.005, i, f"{v*100:.1f}%", va='center')

plt.show()