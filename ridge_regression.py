import pandas as pd 
import datetime as dt
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ridge_regression, Ridge
import matplotlib.pyplot as plt

def load_data():
    global x_train
    global x_validation
    global y_train
    global y_validation
    global y_cols
    
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

try:
    load_data()
except:
    os.system("python3 feature_engineering.py") 
    load_data()

#We want to loop through each y-variable and see which ones are most explainable by our input data. 
results = []
#Taking initial value of alpha = 10 (can change later...)
#First, just training and evaluating on model fitted to training data - just to make our processes really rigorous
alp = float(10)
for column in y_cols: 
    y_train_col = y_train[column]
    y_validation_col = y_validation[column]
    model = Ridge(alp)
    model.fit(x_train, y_train_col)
    y_pred = model.predict(x_validation)
    mse = mean_squared_error(y_validation_col, y_pred)
    rsq = r2_score(y_validation_col, y_pred)
    results.append({'y_column': column, 'MSE': mse, 'R_squared': rsq})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Sort the results by R_squared in descending order
sorted_results = results_df.sort_values(by='R_squared', ascending=False)
best_models = sorted_results[0:70]
# print(sorted_results[1:15])

selected_y_vals = list(best_models["y_column"])
alphas = np.logspace(-4, 5, 100)
fine_tuned_results = []
for column in selected_y_vals: 
    best_rsq = 0
    for alpha in alphas: 
        y_train_col = y_train[column]
        y_validation_col = y_validation[column]
        model = Ridge(alpha)
        model.fit(x_train, y_train_col)
        y_pred = model.predict(x_validation)
        mse = mean_squared_error(y_validation_col, y_pred)
        rsq = r2_score(y_validation_col, y_pred)
        if rsq > best_rsq: 
            best_rsq = rsq
            best_alpha = alpha
    fine_tuned_results.append({'y_column': column, 'Best_Alpha': best_alpha, 'R_squared': best_rsq})
fine_tuned_results = pd.DataFrame(fine_tuned_results).sort_values(by='R_squared', ascending=False)

selected_y_vals = list(fine_tuned_results.iloc[:16, :]["y_column"])

grid_size = 4

fig, axes = plt.subplots(nrows = grid_size, ncols = grid_size)
axes = axes.flatten()
i = 0
for column in selected_y_vals: 
    alp = fine_tuned_results.loc[fine_tuned_results["y_column"]==column, "Best_Alpha"]
    y_train_col = y_train[column]
    y_validation_col = y_validation[column]
    model = Ridge(alp.iloc[0])
    model.fit(x_train, y_train_col)
    y_pred = model.predict(x_validation)
    rsq = r2_score(y_validation_col, y_pred)
    mse = mean_squared_error(y_validation_col, y_pred)
    # Create a subplot for each target variable
    axes[i].scatter(y_validation_col, y_pred, alpha=0.6)
    axes[i].plot(y_validation_col, y_validation_col, color='red', linestyle='--')  # Line of equality
    axes[i].set_title(f'{column} (R² = {rsq:.2f}), (MSE = {mse:.2f})')
    axes[i].set_xlabel('Actual Values')
    axes[i].set_ylabel('Predicted Values')
    i += 1
    
plt.tight_layout()
plt.show()