from Scripts.dataset_reader import get_cpi, get_yield_curve_rates, get_unemployment_rates, get_house_prices, get_gdp, get_labor_productivity
import pandas as pd 
from statsmodels.tsa.api import SimpleExpSmoothing
import os

yield_curve_data = get_yield_curve_rates()

cpi_data = get_cpi()
unemployment = get_unemployment_rates()
house_prices = get_house_prices()
gdp = get_gdp()
labor_productivity = get_labor_productivity()


data_full = pd.merge(cpi_data, unemployment, how = "outer", on = "Date")

data_full = pd.merge(data_full, house_prices, how = "outer", on = "Date")
data_full = pd.merge(data_full, gdp, how="outer", on = "Date")
data_full = pd.merge(data_full, labor_productivity, how = "outer", on = "Date")

# output_path = os.path.join(os.getcwd(), "Cleaned Data")

# data_full.to_csv("joined_file.csv", index = False) # Save the dataset

data_full.ffill(inplace=True)

# output_path = os.path.join(os.getcwd(), "Cleaned Data")

# data_full.to_csv("joined_file_filled.csv", index = False) # Save the dataset

#Value that defines strength of the smoothing 
alpha = 0.2 

for column in data_full.columns: 
    if column != "Date":
        smoothed_col_name = f"{column}_smoothed"
        model = SimpleExpSmoothing(data_full[column])
        data_full = data_full.assign(**{str(smoothed_col_name): (model.fit(smoothing_level = alpha, optimized = False)).fittedvalues})
        column_name = f"{column}_resid"
        data_full = data_full.assign(**{str(column_name): data_full[column] - data_full[smoothed_col_name]})

# output_path = os.path.join(os.getcwd(), "Cleaned Data")

# data_full.to_csv("joined_file_smoothing.csv", index = False) # Save the dataset

look_fwd_windows = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
spreads = [["1 Mo", "3 Mo"], ["1 Mo", "6 Mo"], ["3 Mo", "6 Mo"], ["3 Mo", "3 Yr"],["3 Mo", "1 Yr"], ["1 Yr", "3 Yr"], ["1 Yr", "5 Yr"], ["2 Yr", "10 Yr"], ["1 Yr", "10 Yr"], ["3 Yr", "5 Yr"], ["3 Yr", "20 Yr"], ["3 Yr", "30 Yr"], ["5 Yr", "10 Yr"], ["5 Yr", "20 Yr"], ["5 Yr", "30 Yr"], ["10 Yr", "30 Yr"], ["10 Yr", "20 Yr"]]

for spread in spreads: 
    spread_name = f"{spread[0]}_{spread[1]}_spread"
    yield_curve_data[spread_name] = yield_curve_data[spread[1]]- yield_curve_data[spread[0]]

for column in yield_curve_data.columns: 
    if column != "Date":
        for window in look_fwd_windows: 
            column_name = f"{column}_{window}d_change"
            yield_curve_data = yield_curve_data.assign(**{column_name: yield_curve_data[column].shift(window) - yield_curve_data[column]})

# output_path = os.path.join(os.getcwd(), "Cleaned Data")

# yield_curve_data.to_csv("yield_curve_ftreng.csv", index = False) # Save the dataset

data_full["Date"] = pd.to_datetime(data_full["Date"])
data_full.set_index("Date", inplace=True)
yield_curve_data["Date"] = pd.to_datetime(yield_curve_data["Date"])
yield_curve_data.set_index("Date", inplace=True)

all_dates = pd.date_range(start = "2013-01-01", end = "2024-12-31")

filled_yield_data = yield_curve_data.reindex(all_dates, method='ffill')
filled_yield_data = filled_yield_data.assign(Date = filled_yield_data.index)
filled_yield_data.index = range(len(filled_yield_data.index))

filled_yield_data.insert(0, 'Date', filled_yield_data.pop('Date'))

data_full = data_full.assign(Date = data_full.index)
data_full.index = range(len(data_full.index))

data_full.insert(0, 'Date', data_full.pop('Date'))

all_data = pd.merge(data_full, filled_yield_data, how = "left", on = "Date")

train = all_data[all_data["Date"] < '2020-12-31']

validation = all_data[(all_data["Date"] < '2022-12-31')&(all_data["Date"] > '2020-12-31')]

x_cols = list(data_full.columns)
y_cols = list(filled_yield_data.columns)

train[x_cols].to_csv("training_x_data.csv", index = False)
train[y_cols].to_csv("training_y_data.csv", index = False)

validation[x_cols].to_csv("validation_x_data.csv", index = False)
validation[y_cols].to_csv("validation_y_data.csv", index = False)