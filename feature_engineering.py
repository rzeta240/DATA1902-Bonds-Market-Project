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

data_full.fillna(method = "ffill", inplace = True)


# output_path = os.path.join(os.getcwd(), "Cleaned Data")

# data_full.to_csv("joined_file_filled.csv", index = False) # Save the dataset

#Value that defines strength of the smoothing 
alpha = 0.2 

for column in data_full.columns: 
    if column != "Date":
        smoothed_col_name = f"{column}_smoothed"
        model = SimpleExpSmoothing(data_full[column])
        data_full[smoothed_col_name] = (model.fit(smoothing_level = alpha, optimized = False)).fittedvalues
        data_full[f"{column}_resid"] = data_full[column] - data_full[smoothed_col_name]

output_path = os.path.join(os.getcwd(), "Cleaned Data")

data_full.to_csv("joined_file_smoothing.csv", index = False) # Save the dataset

look_fwd_windows = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
spreads = [["1 Mo", "3 Mo"], ["1 Mo", "6 Mo"], ["3 Mo", "6 Mo"], ["3 Mo", "3 Yr"],["3 Mo", "1 Yr"], ["1 Yr", "3 Yr"], ["1 Yr", "5 Yr"], ["2 Yr", "10 Yr"], ["1 Yr", "10 Yr"], ["3 Yr", "5 Yr"], ["3 Yr", "20 Yr"], ["3 Yr", "30 Yr"], ["5 Yr", "10 Yr"], ["5 Yr", "20 Yr"], ["5 Yr", "30 Yr"], ["10 Yr", "30 Yr"], ["10 Yr", "20 Yr"]]

for spread in spreads: 
    spread_name = f"{spread[0]}_{spread[1]}_spread"
    yield_curve_data[spread_name] = yield_curve_data[spread[1]]- yield_curve_data[spread[0]]

for column in yield_curve_data.columns: 
    if column != "Date":
        for window in look_fwd_windows: 
            column_name = f"{column}_{window}d_change"
            yield_curve_data[column_name] = yield_curve_data[column].shift(-window) - yield_curve_data[column]


# output_path = os.path.join(os.getcwd(), "Cleaned Data")

# yield_curve_data.to_csv("yield_curve_ftreng.csv", index = False) # Save the dataset


