from Scripts.dataset_reader import get_cpi
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import datetime as dt

cpi_data = get_cpi()

cpi_data['Date'] = pd.to_datetime(cpi_data.Date)

# plt.plot(cpi_data.Date, cpi_data.CPIAUCSL_PC1, linewidth = 3)
# plt.title("US CPI (Quarterly)")
# plt.xlabel("Date")
# plt.ylabel("Change in CPI (%)")

# plt.show()
cpi_data["quarter"] = ["Q1" if x.month == 1 else "Q2" if x.month == 4 else "Q3" if x.month == 7 else "Q4" for x in cpi_data['Date']] 
cpi_yearly_average = cpi_data.groupby(cpi_data["Date"].dt.year)['CPIAUCSL_PC1'].mean().reset_index()

cpi_yearly_average.columns = ["Year", "Average"]

cpi_data = cpi_data.merge(cpi_yearly_average, left_on = cpi_data["Date"].dt.year, right_on="Year", how = 'left')

cpi_data["diff_from_avg"] = cpi_data['CPIAUCSL_PC1'] - cpi_data['Average']

print(cpi_data)

sns.boxplot(x = "quarter", y = "diff_from_avg", data = cpi_data)
plt.show()