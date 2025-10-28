import pandas as pd 
import datetime as dt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ridge_regression, Ridge
import matplotlib.pyplot as plt


eco_data = pd.read_csv("joined_file_smoothing.csv")
yield_data = pd.read_csv("yield_curve_ftreng.csv")

eco_data["Date"] = pd.to_datetime(eco_data["Date"])
eco_data.set_index("Date", inplace=True)
yield_data["Date"] = pd.to_datetime(yield_data["Date"])
yield_data.set_index("Date", inplace=True)
y_cols = list(yield_data.columns)
x_cols = list(eco_data.columns)

all_dates = pd.date_range(start = "2013-01-01", end = "2024-12-31")

#Forward fills any missing dates in YC data to ensure there are predictors for every Eco Data input (when the first of the month falls on a weekday)
filled_yield_data = yield_data.reindex(all_dates, method='ffill')
filled_yield_data = filled_yield_data.assign(Date = filled_yield_data.index)
filled_yield_data.index = range(len(filled_yield_data.index))

filled_yield_data.insert(0, 'Date', filled_yield_data.pop('Date'))

all_data = pd.merge(eco_data, filled_yield_data, how = "left", on = "Date")

#split into Training, validation, test - 8 yrs for training, 2 for validation, 2 for test
train = all_data[all_data["Date"] < '2020-12-31']
validation = all_data[(all_data["Date"] < '2022-12-31')&(all_data["Date"] > '2020-12-31')]
#We want to loop through each y-variable and see which ones are most explainable by our input data. 
results = []
#Taking initial value of alpha = 10 (can change later...)
#First, just training and evaluating on model fitted to training data - just to make our processes really rigorous
x_train = train[x_cols]
x_validate = validation[x_cols]
alp = 10
for column in y_cols: 
    y_train = train[column]
    y_validate = validation[column]
    model = Ridge(alp)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_validate)
    mse = mean_squared_error(y_validate, y_pred)
    rsq = r2_score(y_validate, y_pred)
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
        y_train = train[column]
        y_validate = validation[column]
        model = Ridge(alpha)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_validate)
        mse = mean_squared_error(y_validate, y_pred)
        rsq = r2_score(y_validate, y_pred)
        if rsq > best_rsq: 
            best_rsq = rsq
            best_alpha = alpha
    fine_tuned_results.append({'y_column': column, 'Best_Alpha': best_alpha, 'R_squared': best_rsq})
fine_tuned_results = pd.DataFrame(fine_tuned_results).sort_values(by='R_squared', ascending=False)



selected_y_vals = list(fine_tuned_results[10:26]["y_column"])

grid_size = 4

fig, axes = plt.subplots(nrows = grid_size, ncols = grid_size)
axes = axes.flatten()
i = 0
for column in selected_y_vals: 
    alp = fine_tuned_results.loc[fine_tuned_results["y_column"]==column, "Best_Alpha"]
    y_train = train[column]
    y_validate = validation[column]
    model = Ridge(float(alp))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_validate)
    rsq = r2_score(y_validate, y_pred)
    mse = mean_squared_error(y_validate, y_pred)
    # Create a subplot for each target variable
    axes[i].scatter(y_validate, y_pred, alpha=0.6)
    axes[i].plot(y_validate, y_validate, color='red', linestyle='--')  # Line of equality
    axes[i].set_title(f'{column} (R² = {rsq:.2f}), (MSE = {mse:.2f})')
    axes[i].set_xlabel('Actual Values')
    axes[i].set_ylabel('Predicted Values')
    i += 1
plt.tight_layout()
plt.show()