import pandas as pd
import numpy as np
from datetime import datetime
import os

datasets_path = os.path.join(os.getcwd(), "Datasets")

def date(y, m, d):
    return datetime.date(datetime(y, m, d))

d_0 = date(2013, 1, 1)
d_f = date(2024, 12, 31)

def percent_present(data_name, df):
    cols = df.columns

    data_name_lens = [4] * len(cols)
    for i in range( len( cols ) ):
        data_name_lens[i] = max( len( cols[i] ), 4 )

    percent_data_present = []

    for i in range(len(df.columns)):
        c = np.array(df.iloc[:, i])

        filter = []

        for j in range(len( c ) ):
            filter.append(True)

            if type( c[j] ) == str:
                filter[j] = False
                continue

            if type( c[j] ) == np.float64:
                if np.isnan( c[j] ):
                    filter[j] = False

        c_filtered = c[filter]

        percent_data_present.append(f"{len(c_filtered) / len(c) * 100:.0f}%")

    print(f"{data_name} Data Loaded.\nPercent of data present:")
    print(" | ".join([f"{df.columns[i].center(data_name_lens[i])}" for i in range( len( percent_data_present ) )]))
    print(" | ".join([f"{percent_data_present[i].center(data_name_lens[i])}" for i in range( len( percent_data_present ) )]))


def get_yield_curve_rates(loadmsg = False):
    df = pd.read_csv(os.path.join(datasets_path, "yield_curve_rates_daily.csv"))

    for i in range( len( df["Date"] ) ):
        date_str = df["Date"][i]

        date_obj = datetime.date(datetime.strptime(date_str, "%m/%d/%Y"))
        
        df.at[i, "Date"] = date_obj

    df = df[df["Date"] >= d_0]
    df = df[df["Date"] <= d_f]
    
    df.pop("2 Mo")
    df.pop("4 Mo")

    if loadmsg:
        percent_present("Yield Curve", df)

    df.sort_values("Date")

    return df

def get_unemployment_rates(loadmsg = False):
    df = pd.read_csv(os.path.join(datasets_path, "unemployment_rate_monthly.csv"))

    for i in range( len( df["Date"] ) ):
        date_str = df["Date"][i]

        date_obj = datetime.date(datetime.strptime(date_str, "%Y-%m-%d"))
        
        df.at[i, "Date"] = date_obj

    df = df[df["Date"] >= d_0]
    df = df[df["Date"] <= d_f]

    if loadmsg:
        percent_present("Unemployment", df)

    df.sort_values("Date")

    return df

def get_cpi(loadmsg = False):
    df = pd.read_csv(os.path.join(datasets_path, "consumer_price_index_quarterly.csv"))

    for i in range( len( df["Date"] ) ):
        date_str = df["Date"][i]

        date_obj = datetime.date(datetime.strptime(date_str, "%m/%d/%Y"))
        
        df.at[i, "Date"] = date_obj

    df = df[df["Date"] >= d_0]
    df = df[df["Date"] <= d_f]

    if loadmsg:
        percent_present("CPI (Inflation)", df)

    df.sort_values("Date")

    return df

def get_house_prices(loadmsg = False):
    df = pd.read_csv(os.path.join(datasets_path, "average_house_price_quarterly.csv"))

    for i in range( len( df["Date"] ) ):
        date_str = df["Date"][i]

        date_obj = datetime.date(datetime.strptime(date_str, "%Y-%m-%d"))
        
        df.at[i, "Date"] = date_obj

    df = df[df["Date"] >= d_0]
    df = df[df["Date"] <= d_f]

    if loadmsg:
        percent_present("House Price", df)

    df.sort_values("Date")

    return df

def get_labor_productivity(loadmsg = False):
    df = pd.read_csv(os.path.join(datasets_path, "labor_productivity_quarterly.csv"))

    df = df[df["Units"] == "% Change from previous quarter"]

    df.pop("Basis")
    df.pop("Units")

    sector = list(df["Sector"])
    measure = list(df["Measure"])

    sector_and_measure = []

    for i in range( len( sector ) ):
        sector_and_measure.append(f"{sector[i]} {measure[i]}")
    
    values = df.loc[:, "1947 Q1":]
    values.columns = range(len(values.columns))

    values = values.transpose()
    values.columns = sector_and_measure

    quarters = df.columns[2:]

    dates = {"Date": []}

    for q in quarters:
        y = int(q[:4])
        qi = int(q[6:])

        dates["Date"].append(date(y, 3*qi - 2, 1))

    dates_df = pd.DataFrame(dates)

    df = pd.concat([dates_df, values], axis = 1)

    df = df[df["Date"] >= d_0]
    df = df[df["Date"] <= d_f]
    df.index = range(len(df.index))

    def fix(col):
        if col.name == "Date":
            return col
        
        newcol = []

        for x in col:
            try:
                newcol.append(np.float64(x))
            except:
                newcol.append(float('nan'))

        return newcol

    df = df.apply(fix)

    for col in df.columns:
        if df[col][0] != df[col][0]:
            df.pop(col)

    return df

def get_gdp(loadmsg = False):
    df = pd.read_csv(os.path.join(datasets_path, "GDP_quarterly.csv"))

    for i in range( len( df["Date"] ) ):
        date_str = df["Date"][i]

        date_obj = datetime.date(datetime.strptime(date_str, "%Y-%m-%d"))
        
        df.at[i, "Date"] = date_obj

    df = df[df["Date"] >= d_0]
    df = df[df["Date"] <= d_f]

    if loadmsg:
        percent_present("GDP", df)

    df.sort_values("Date")

    return df

if __name__ == "__main__":
    yield_rates = get_yield_curve_rates()
    un_rates = get_unemployment_rates()
    cpi = get_cpi()
    house_prices = get_house_prices()
    labor_productivity = get_labor_productivity()
    gdp = get_gdp()

    output_path = os.path.join(os.getcwd(), "Cleaned Data")
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    dfs = [yield_rates, 
           un_rates, 
           cpi, 
           house_prices, 
           labor_productivity,
           gdp
           ]
    names = ["yield_curve_rates_daily_2013_2024.csv", 
             "unemployment_rate_monthly.csv", 
             "consumer_price_index_quarterly.csv", 
             "average_house_price_quarterly.csv", 
             "labor_productivity_quarterly.csv",
             "GDP_quarterly.csv"
             ]

    for i in range(len(dfs)):
        df = dfs[i]
        name = names[i]

        for i in range(len(df["Date"])):
            df.at[i, "Date"] = df.loc[i, "Date"].strftime("%d/%m/%Y")
        
        df.to_csv(os.path.join(output_path, name), index = False)