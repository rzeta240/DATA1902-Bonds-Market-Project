import pandas as pd
import matplotlib.pyplot as plt

from Scripts.dataset_reader import get_cpi, get_gdp, get_house_prices, get_unemployment_rates

names = ["Consumer Price Index",
         "Gross Domestic Product",
         "Average House Price",
         "Unemployment Rate"]

choice = input( # We print out the names of the datasets for the user to select. 
    "Select dataset(s) to visualise:\n" +
    "\n".join(f"[{i+1}]: {names[i]}" for i in range( len( names ) )) +
    "\n"
)
choices = [int(i)-1 for i in list(choice)] # The user has inputted a string of numbers. We split these and converts them into ints

df_readers = [[get_cpi, get_gdp, get_house_prices, get_unemployment_rates][i] for i in choices] # Select the reader functions corresponding to the user's choices

dfs = [df_reader() for df_reader in df_readers] # Run the reader functions to get the dataframes

ax = plt.gca() # Get current axes. Used for setting y-axis limit

if len(dfs) == 1: # If they only want one dataframe
    df = dfs[0]
    
    plt.plot(df["Date"], df.iloc[:, 1]) # We plot it
    plt.ylabel(df.columns[1])
    plt.title(names[choices[0]])
    
    v = max(df.iloc[:, 1]) # Find the max value
    
    ax.set_ylim( [-v*0.1, v*1.1] ) # Set the y-axis to show all values, and down to 0
else:
    for df in dfs: # For every dataframe
        df = df.assign(normalised = df.iloc[:, 1] / max(df.iloc[:, 1])) # We normalise the data
        
        plt.plot(df["Date"], df.loc[:, "normalised"], label = df.columns[1]) # And plot that

    plt.title(f"Normalised graphs over time of {", ".join([ names[i] for i in choices ])}", wrap=True)
    
    ax.set_ylim( [-0.1, 1.1] ) # Make sure it fits

    plt.legend() # Add legend
    
plt.xlabel("Date") 
plt.show()