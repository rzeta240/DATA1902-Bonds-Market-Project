import pandas as pd 
from sklearn import metrics
from sklearn.linear_model import ridge_regression


eco_data = pd.read_csv("joined_file_smoothing.csv")
yield_data = pd.read_csv("yield_curve_ftreng.csv")

eco_data["Date"] = pd.to_datetime(eco_data["Date"])
eco_data.set_index("Date", inplace=True)
y_cols = list(yield_data.columns)
x_cols = list(eco_data.columns)

all_dates = pd.date_range(start = "2013-01-01", end = "2024-12-31")


#Forward fills any missing dates in YC data to ensure there are predictors for every Eco Data input (when the first of the month falls on a weekday)
filled_eco_data = eco_data.reindex(all_dates)
filled_eco_data = filled_eco_data.ffill()



all_data = pd.merge(filled_eco_data, yield_data, how = "left", on = "Date")
print(all_data)

#We want to loop through each y-variable and see which ones are most explainable by our input data. 
