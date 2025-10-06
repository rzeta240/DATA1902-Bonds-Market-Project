from Scripts.dataset_reader import get_cpi, get_yield_curve_rates
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import datetime as dt

cpi_data = get_cpi()

yield_curve_rates = get_yield_curve_rates()

cpi_data['Date'] = pd.to_datetime(cpi_data.Date)

# plt.plot(cpi_data.Date, cpi_data.CPIAUCSL_PC1, linewidth = 3)
# plt.title("US CPI (Quarterly)")
# plt.xlabel("Date")
# plt.ylabel("Change in CPI (%)")

# # plt.show()


# cpi_data["quarter"] = ["Q1" if x.month == 1 else "Q2" if x.month == 4 else "Q3" if x.month == 7 else "Q4" for x in cpi_data['Date']] 
# cpi_yearly_average = cpi_data.groupby(cpi_data["Date"].dt.year)['CPIAUCSL_PC1'].mean().reset_index()

# cpi_yearly_average.columns = ["Year", "Average"]

# cpi_data = cpi_data.merge(cpi_yearly_average, left_on = cpi_data["Date"].dt.year, right_on="Year", how = 'left')

# cpi_data["diff_from_avg"] = cpi_data['CPIAUCSL_PC1'] - cpi_data['Average']

# print(cpi_data)

# sns.boxplot(x = "quarter", y = "diff_from_avg", data = cpi_data)
# plt.show()
cpi_data['month'] = cpi_data['Date'].dt.month
yield_curve_rates['month'] = pd.to_datetime(yield_curve_rates['Date']).dt.month

yield_curve_rates["frontend_micro_spread"] = yield_curve_rates['6 Mo'] - yield_curve_rates['3 Mo']
yield_curve_rates["frontend_macro_spread"] = yield_curve_rates['3 Yr'] - yield_curve_rates['1 Yr'] 

#Calculating 10-day forward change in spreads. Note that we are interested in future changes and not historical changes as the ultimate goal is to predict future changes in
#these spreads to build a profitable trading strategy
yield_curve_rates["10d_change_micro_spread"] = yield_curve_rates['frontend_micro_spread'].shift(-10) - yield_curve_rates['frontend_micro_spread']

yield_curve_rates["10d_change_macro_spread"] = yield_curve_rates['frontend_macro_spread'].shift(-10) - yield_curve_rates['frontend_macro_spread']

new_df = pd.merge(yield_curve_rates, cpi_data, on='month', how = 'left')

print(new_df)
plt.scatter(new_df["CPIAUCSL_PC1"], new_df["10d_change_macro_spread"])
plt.show()