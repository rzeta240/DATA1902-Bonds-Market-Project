import pandas as pd
import numpy as np
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

start = time.time()

# Load training and validation datasets
def load_data():
    global x_train, x_val, y_train, y_val, y_cols

    x_train = pd.read_csv("training_x_data.csv")
    x_train["Date"] = pd.to_datetime(x_train["Date"])
    x_train.set_index("Date", inplace=True)

    x_val = pd.read_csv("validation_x_data.csv")
    x_val["Date"] = pd.to_datetime(x_val["Date"])
    x_val.set_index("Date", inplace=True)

    y_train = pd.read_csv("training_y_data.csv")
    y_train["Date"] = pd.to_datetime(y_train["Date"])
    y_train.set_index("Date", inplace=True)

    y_val = pd.read_csv("validation_y_data.csv")
    y_val["Date"] = pd.to_datetime(y_val["Date"])
    y_val.set_index("Date", inplace=True)

    y_cols = y_train.columns

try:
    load_data()
except:
    os.system("python3 feature_engineering.py")
    load_data()

# Initialize storage for model evaluation results
results = []

# Train one random forest per target and evaluate on validation window
for col in tqdm(y_cols, desc="Training RF Models", ncols=90):
    y_train_col = y_train[col]
    y_val_col = y_val[col]

    model = RandomForestRegressor(
        n_estimators=300,      # number of trees
        random_state=42,       # reproducibility
        n_jobs=-1              # use all CPU cores
    )

    model.fit(x_train, y_train[col])         # train model on 2013–2020
    y_pred = model.predict(x_val)            # validate on 2021–2022

    r2 = r2_score(y_val[col], y_pred)        # R² score on validation

    results.append({
        "target": col,
        "R2": r2
    })

# Rank targets and select the top-16 performers
results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
top16 = results_df.head(16)
best_targets = list(top16["target"])

# Save selected spreads for later out-of-sample testing
top16.to_csv("rf_selected_targets.csv", index=False)

print("Top 16 spreads selected and saved to rf_selected_targets.csv")
print(top16)

# scatter plot aesthetics
colors = ["#7db9e8", "#a18cd1", "#fbc2eb", "#f6d365", "#fda085"]
cmap = LinearSegmentedColormap.from_list("soft_grad", colors)

# Plot scatter charts for top-16 spreads
fig, axes = plt.subplots(4, 4, figsize=(15,10))
axes = axes.flatten()

for i, col in enumerate(best_targets):
    model = RandomForestRegressor(
        n_estimators=300,      # number of trees
        random_state=42,       # reproducibility
        n_jobs=-1              # use all CPU cores
    )
    model.fit(x_train, y_train[col])
    y_pred = model.predict(x_val)

    actual = y_val[col]
    norm = (actual - actual.min()) / (actual.max() - actual.min())

    axes[i].scatter(
        actual, 
        y_pred,
        marker="^",
        c=cmap(norm),
        s=48,
        alpha=0.75,
        linewidths=0
    )

    axes[i].plot(actual, actual, color="#e57373", linestyle="--", linewidth=1.6, alpha=0.9)

    axes[i].set_xlabel("Actual Yield Change")
    axes[i].set_ylabel("Predicted Yield Change")

    axes[i].set_title(f"{col}\nR²={r2_score(actual, y_pred):.2f}", fontsize=9)
    axes[i].tick_params(labelsize=8)

plt.tight_layout()
plt.show()

end = time.time()
print(f"Training and visualization completed in {end - start:.2f} seconds")