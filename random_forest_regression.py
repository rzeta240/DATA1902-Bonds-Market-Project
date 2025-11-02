import pandas as pd
import numpy as np
import os, time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

start = time.time()


# Load dataset and convert Date to index for time-series use
def load_data():
    global x_train, x_val, y_train, y_val, y_cols
    
    # Load training and validation feature sets
    x_train = pd.read_csv("training_x_data.csv").set_index("Date")
    x_train.index = pd.to_datetime(x_train.index)

    x_val = pd.read_csv("validation_x_data.csv").set_index("Date")
    x_val.index = pd.to_datetime(x_val.index)

    # Load training and validation target (yield spread changes)
    y_train = pd.read_csv("training_y_data.csv").set_index("Date")
    y_train.index = pd.to_datetime(y_train.index)

    y_val = pd.read_csv("validation_y_data.csv").set_index("Date")
    y_val.index = pd.to_datetime(y_val.index)

    # Store target column names (each spread is a target)
    y_cols = y_train.columns[11:]

# run feature engineering
try:
    load_data()
except:
    os.system("python3 feature_engineering.py")
    load_data()


# Train RF for each yield spread & rank by validation R²
results = []
models = {}

for col in tqdm(y_cols, desc="Training RF Models", ncols=90):
    # Train one random forest per target spread
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train[col])

    models[col] = model

    # Predict on validation window (2021–2022)
    pred = model.predict(x_val)
    
    # Compute R² accuracy to rank models
    r2 = r2_score(y_val[col], pred)
    mse = mean_squared_error(y_val[col], pred)
    
    results.append({"target": col, "R2": r2, "MSE": mse})

# Select the top 16 spreads by predictive performance
results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)

print(results_df.head(16))

best_targets = list(results_df.head(16)["target"])

# Print validation R² and MSE for the top 16
print("\nValidation Metrics for Top 16:")
print(results_df.head(16)[["target", "R2", "MSE"]])

# Train final RF models for the winning spreads
# for col in best_targets:
#     m = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
#     m.fit(x_train, y_train[col])
#     models[col] = m


# Load test window (2023–2024) for out-of-sample evaluation
test_x = pd.read_csv("test_x_data.csv").set_index("Date")
test_y = pd.read_csv("test_y_data.csv").set_index("Date")
test_x.index = pd.to_datetime(test_x.index)
test_y.index = pd.to_datetime(test_y.index)

pnl_results = {}
hit_results = {}

print("\nTest Set Metrics for Top 16:")

# Convert model forecasts into trading positions & P&L

for col in best_targets:
    
    # Only use rows where target has valid values
    idx = ~test_y[col].isna() # ~ is the logical NOT operator for pandas boolean arrays.
    X = test_x.loc[idx]
    y = test_y.loc[idx, col]

    # Predict yield change for test window
    pred = models[col].predict(X)

    # Evaluate prediction accuracy on test set
    rsq = r2_score(y, pred)
    mse = mean_squared_error(y, pred)
    print(f"{col} — Test R²: {rsq:.4f}, MSE: {mse:.4f}")

    pred1 = models[col].predict(x_train)
    pred2 = models[col].predict(x_val)

    # Trading rule:
    # If predicted yields rise then short bonds, else long bonds
    signal = np.where(pred > 0, 1, -1)

    # Scale position by prediction confidence (size between 0.3 and 1)
    scale = 5 * np.abs(pred) / (np.mean(list(np.abs(pred1)) + list(np.abs(pred2))) + 1e-6)
    size = np.clip(scale, 0.3, 1.0)

    # Convert yield move into P&L in basis points ($100 per bp)
    pnl = -signal * size * y * 100
    pnl_results[col] = pnl

    # Compute directional accuracy (hit rate)
    hit_results[col] = np.mean((signal > 0) == (y < 0))


# Sort spreads by profitability for clean visual ranking
sorted_targets = sorted(best_targets, key=lambda c: pnl_results[c].sum(), reverse=True)


# Plot cumulative P&L for each spread (our trading performance)
colors = ["#7db9e8","#a18cd1","#fbc2eb","#f6d365","#fda085"]
cmap = LinearSegmentedColormap.from_list("grad", colors)

fig, axes = plt.subplots(4,4, figsize=(15,10))
axes = axes.flatten()

for i, col in enumerate(sorted_targets):
    
    pnl = pnl_results[col]
    cum = np.cumsum(pnl)
    
    # Color intensity based on profit trend
    norm = (cum - cum.min()) / (cum.max() - cum.min() + 1e-6)

    # Plot cumulative returns curve 
    axes[i].plot(cum.index, cum, color="#7db9e8", linewidth=2)
    axes[i].scatter(cum.index, cum, c=cmap(norm), s=35, alpha=0.9, marker="^")
    axes[i].axhline(0, color="#e57373", linestyle="--")

    axes[i].set_title(
        f"Spread: {col}\nTotal Cumulative Profit: ${cum.iloc[-1]:,.0f}\nDirectional Accuracy: {hit_results[col]*100:.1f}%",
        fontsize=8
    )
    axes[i].tick_params(labelsize=7)

plt.tight_layout()
plt.show()

print("\nDone in", round(time.time()-start,2), "sec")