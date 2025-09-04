import pandas as pd
import numpy as np
from datetime import datetime
import os
import math

datasets_path = os.path.join(os.getcwd(), "Datasets")

yield_curve_rates = pd.read_csv("Datasets/yield_curve_rates_daily.csv")

def date(y, m, d):
    return datetime.date(datetime(y, m, d))

def get_yield_curve_rates():
    df = pd.read_csv(os.path.join(datasets_path, "yield_curve_rates_daily.csv"))

    for i in range( len( df["Date"] ) ):
        date_str = df["Date"][i]

        date_obj = datetime.date(datetime.strptime(date_str, "%m/%d/%Y"))
        
        df.at[i, "Date"] = date_obj

    df = df[df["Date"] >= date(2013, 1, 1)]

    # df.rename(columns={"1 Mo": 1, 
    #                    "2 Mo": 2,
    #                    "3 Mo": 3,
    #                    "4 Mo": 4,
    #                    "6 Mo": 6,
    #                    "1 Yr": 12,
    #                    "2 Yr": 24,
    #                    "3 Yr": 36,
    #                    "5 Yr": 60,
    #                    "7 Yr": 84,
    #                    "10 Yr": 120,
    #                    "20 Yr": 240,
    #                    "30 Yr": 360
    #                    })
    
    percent_data_present = []

    for i in range(len(df.columns)):
        c = list(df.iloc[:, i])

        for j in range( len( c ) ):
            if type(c[j]) == float:
                if np.isnan(c[j]):
                    c[j] = None
        
        c_filtered = list(filter(None, c))

        percent_data_present.append(f"{len(c_filtered) / len(c) * 100:.0f}%")

    print("Yield Curve Data Loaded.\nPercent of data present:")
    print(" | ".join([f"{df.columns[i]:^6}" for i in range( len( percent_data_present ) )]))
    print(" | ".join([f"{percent_data_present[i]:^6}" for i in range( len( percent_data_present ) )]))

    df.sort_values("Date")

    return df

if __name__ == "__main__":
    get_yield_curve_rates()