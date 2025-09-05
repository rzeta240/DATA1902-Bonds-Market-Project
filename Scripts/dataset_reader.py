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


def get_yield_curve_rates(silent = False):
    df = pd.read_csv(os.path.join(datasets_path, "yield_curve_rates_daily.csv"))

    for i in range( len( df["Date"] ) ):
        date_str = df["Date"][i]

        date_obj = datetime.date(datetime.strptime(date_str, "%m/%d/%Y"))
        
        df.at[i, "Date"] = date_obj

    df = df[df["Date"] >= d_0]
    df = df[df["Date"] <= d_f]
    
    if not silent:
        percent_present("Yield Curve", df)

    df.sort_values("Date")

    return df

def get_unemployment_rates(silent = False):
    df = pd.read_csv(os.path.join(datasets_path, "unemployment_rate_monthly.csv"))

    for i in range( len( df["Date"] ) ):
        date_str = df["Date"][i]

        date_obj = datetime.date(datetime.strptime(date_str, "%Y-%m-%d"))
        
        df.at[i, "Date"] = date_obj

    df = df[df["Date"] >= d_0]
    df = df[df["Date"] <= d_f]

    if not silent:
        percent_present("Unemployment", df)

    df.sort_values("Date")

    return df

def get_cpi(silent = False):
    df = pd.read_csv(os.path.join(datasets_path, "consumer_price_index_quarterly.csv"))

    for i in range( len( df["Date"] ) ):
        date_str = df["Date"][i]

        date_obj = datetime.date(datetime.strptime(date_str, "%m/%d/%Y"))
        
        df.at[i, "Date"] = date_obj

    df = df[df["Date"] >= d_0]
    df = df[df["Date"] <= d_f]

    if not silent:
        percent_present("CPI (Inflation)", df)

    df.sort_values("Date")

    return df

def get_house_prices(silent = False):
    df = pd.read_csv(os.path.join(datasets_path, "average_house_price_quarterly.csv"))

    for i in range( len( df["Date"] ) ):
        date_str = df["Date"][i]

        date_obj = datetime.date(datetime.strptime(date_str, "%Y-%m-%d"))
        
        df.at[i, "Date"] = date_obj

    df = df[df["Date"] >= d_0]
    df = df[df["Date"] <= d_f]

    if not silent:
        percent_present("House Price", df)

    df.sort_values("Date")

    return df

if __name__ == "__main__":
    yield_rates = get_yield_curve_rates()
    un_rates = get_unemployment_rates()
    cpi = get_cpi()
    house_prices = get_house_prices()

    output_path = os.path.join(os.getcwd(), "Cleaned Data")

    for i in range(len(yield_rates["Date"])):
        yield_rates.at[i, "Date"] = yield_rates.loc[i, "Date"].strftime("%d/%m/%Y")
    
    yield_rates.pop("2 Mo")
    yield_rates.pop("4 Mo")

    yield_rates.to_csv(os.path.join(output_path, "yield_curve_rates_daily_2013_2024.csv"), index = False)

    for i in range(len(un_rates["Date"])):
        un_rates.at[i, "Date"] = un_rates.loc[i, "Date"].strftime("%d/%m/%Y")
    
    un_rates.to_csv(os.path.join(output_path, "unemployment_rate_monthly.csv"), index = False)

    for i in range(len(cpi["Date"])):
        cpi.at[i, "Date"] = cpi.loc[i, "Date"].strftime("%d/%m/%Y")
    
    cpi.to_csv(os.path.join(output_path, "consumer_price_index_quarterly.csv"), index = False)

    for i in range(len(house_prices["Date"])):
        house_prices.at[i, "Date"] = house_prices.loc[i, "Date"].strftime("%d/%m/%Y")
    
    house_prices.to_csv(os.path.join(output_path, "average_house_price_quarterly.csv"), index = False)