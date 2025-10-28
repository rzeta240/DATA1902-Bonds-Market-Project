import pandas as pd 
import datetime as dt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ridge_regression, Ridge


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
validation = all_data[(all_data["Date"] < '2022-12-31')|(all_data["Date"] > '2020-12-31')]
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

# Print the y-values with the highest R_squared
print(sorted_results)
