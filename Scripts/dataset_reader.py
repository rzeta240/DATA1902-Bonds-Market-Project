import pandas as pd
import numpy as np
from datetime import datetime
import os

datasets_path = os.path.join(os.getcwd(), "Datasets")

def date(y, m, d): # Saves us writing date 3 times
    return datetime.date(datetime(y, m, d))

d_0 = date(2013, 1, 1) # Start date
d_f = date(2024, 12, 31) # Final date

def percent_present(data_name, df): # Determine how much data in each column is present
    cols = df.columns

    data_name_lens = [4] * len(cols) 
    for i in range( len( cols ) ):
        data_name_lens[i] = max( len( cols[i] ), 4 ) # For our print() output, we don't want to format the columns any smaller than 4 chars, otherwise "100%" won't fit

    percent_data_present = []

    for i in range(len(df.columns)):
        c = np.array(df.iloc[:, i])

        filter_arr = []

        for j in range(len( c ) ):
            filter_arr.append(True)

            if type( c[j] ) == str:
                filter_arr[j] = False # Filter out strings
                continue

            if type( c[j] ) == np.float64:
                if np.isnan( c[j] ): 
                    filter_arr[j] = False # Filter out NaNs 

        c_filtered = c[filter_arr] 

        percent_data_present.append(f"{len(c_filtered) / len(c) * 100:.0f}%") # See what's left

    print(f"{data_name} Data Loaded.\nPercent of data present:") # F-string formatting + list comprehension hell below
    print(" | ".join([f"{df.columns[i].center(data_name_lens[i])}" for i in range( len( percent_data_present ) )]))
    print(" | ".join([f"{percent_data_present[i].center(data_name_lens[i])}" for i in range( len( percent_data_present ) )]))

def load_and_format_dates(df, format): # Read the dates in a csv as a certain format and convert to a datetime.date object
    for i in range( len( df["Date"] ) ):
        date_str = df["Date"][i] # Read

        date_obj = datetime.date(datetime.strptime(date_str, format)) # Convert using the format
        
        df.at[i, "Date"] = date_obj # Replace original

    df = df[df["Date"] >= d_0] # We don't want anything before start date
    df = df[df["Date"] <= d_f] # And nothing after end date

    return df

def get_yield_curve_rates(loadmsg = False): # Load the yield curve rates dataset
    df = pd.read_csv(os.path.join(datasets_path, "yield_curve_rates_daily.csv"))

    df = load_and_format_dates(df, "%m/%d/%Y")
    
    df.pop("2 Mo") # Missing 48% of values
    df.pop("4 Mo") # Missing 82% of values

    if loadmsg:
        percent_present("Yield Curve", df)

    df.sort_values("Date")

    return df

def get_unemployment_rates(loadmsg = False): # Load the unemployment rates dataset
    df = pd.read_csv(os.path.join(datasets_path, "unemployment_rate_monthly.csv"))

    df = load_and_format_dates(df, "%Y-%m-%d")

    if loadmsg:
        percent_present("Unemployment", df)

    df.sort_values("Date")

    return df

def get_cpi(loadmsg = False): # Load the CPI (Inflation) dataset
    df = pd.read_csv(os.path.join(datasets_path, "consumer_price_index_quarterly.csv"))

    df = load_and_format_dates(df, "%m/%d/%Y")

    if loadmsg:
        percent_present("CPI (Inflation)", df)

    df.sort_values("Date")

    return df

def get_house_prices(loadmsg = False): # Load the House prices dataset
    df = pd.read_csv(os.path.join(datasets_path, "average_house_price_quarterly.csv"))
    
    df = load_and_format_dates(df, "%Y-%m-%d")

    if loadmsg:
        percent_present("House Price", df)

    df.sort_values("Date")

    return df

def get_gdp(loadmsg = False): # Load the GSP dataset
    df = pd.read_csv(os.path.join(datasets_path, "GDP_quarterly.csv"))

    df = load_and_format_dates(df, "%Y-%m-%d")

    if loadmsg:
        percent_present("GDP", df)

    df.sort_values("Date")

    return df

def get_labor_productivity(loadmsg = False): # Load the labor productivity dataset
    df = pd.read_csv(os.path.join(datasets_path, "labor_productivity_quarterly.csv"))

    df = df[df["Units"] == "% Change from previous quarter"] # Most easily analysable metric. All others are total or % of 2017 and are less normalised

    df.pop("Basis") # Useless
    df.pop("Units") # Useless since we deleted everything that wasn't % change

    sector = list(df["Sector"])
    measure = list(df["Measure"])

    sector_and_measure = []

    for i in range( len( sector ) ):
        sector_and_measure.append(f"{sector[i]} {measure[i]}") #Combine sector and measure columns into one
    
    values = df.loc[:, "1947 Q1":] # Take all the numerical data
    values.columns = range(len(values.columns)) # Rename the columns because we're about to transpose and the columns will become the index

    values = values.transpose() # Usual matrix transpose
    values.columns = sector_and_measure # New column names (the old row names)

    quarters = df.columns[2:] 

    dates = {"Date": []}

    for q in quarters:
        y = int(q[:4])
        qi = int(q[6:])

        dates["Date"].append(date(y, 3*qi - 2, 1)) # Take the old quarter column names and turn them into datetime.date objects

    dates_df = pd.DataFrame(dates)

    df = pd.concat([dates_df, values], axis = 1) # Add these dates to the values DataFrame

    df = df[df["Date"] >= d_0] # Nothing before start date
    df = df[df["Date"] <= d_f] # Nothing after end date
    df.index = range(len(df.index)) # Make sure the index isn't being tricky

    def fix(col): # Converting values into appropriate data types
        if col.name == "Date":
            return col # Don't touch the dates
        
        newcol = []

        for x in col:
            try:
                newcol.append(np.float64(x)) # Try converting to float64
            except:
                newcol.append(float('nan')) # If failed, call it NaN

        return newcol

    df = df.apply(fix) # Apply fix to all columns

    for col in df.columns:
        if df[col][0] != df[col][0]: # Using percent_present, some columns simply have no data. We pop these
            df.pop(col)

    return df

if __name__ == "__main__":
    yield_rates = get_yield_curve_rates()
    un_rates = get_unemployment_rates()
    cpi = get_cpi()
    house_prices = get_house_prices()
    labor_productivity = get_labor_productivity()
    gdp = get_gdp()

    output_path = os.path.join(os.getcwd(), "Cleaned Data")
    if not os.path.isdir(output_path): # If no directory, make one
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

    for i in range(len(dfs)): # Loop through each loaded dataset and save it under Cleaned Data
        df = dfs[i]
        name = names[i]

        for i in range(len(df["Date"])):
            df.at[i, "Date"] = df.loc[i, "Date"].strftime("%d/%m/%Y") # Convert datetime.date objects into dd-mm-yyyy format
        
        df.to_csv(os.path.join(output_path, name), index = False) # Save the dataset