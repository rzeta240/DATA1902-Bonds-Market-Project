import numpy as np, pandas as pd
from pathlib import Path
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Div, Slider, NumeralTickFormatter
from bokeh.plotting import figure
from bokeh.server.server import Server

def modify_doc(doc):
    #load all data

    #load yield curve data
    df = pd.read_csv("Datasets/yield_curve_rates_daily.csv")

    #load cpi data
    cpi = pd.read_csv("Datasets/consumer_price_index_quarterly.csv")
    cpi["Date"] = pd.to_datetime(cpi["Date"])
    cpi = cpi.sort_values("Date").reset_index(drop=True)

    #Load unemployment data
    df_unemp = pd.read_csv("Datasets/unemployment_rate_monthly.csv")
    df_unemp["Date"] = pd.to_datetime(df_unemp["Date"])
    df_unemp = df_unemp.sort_values("Date").reset_index(drop=True)

    # load gdp data
    df_gdp = pd.read_csv("Datasets/GDP_quarterly.csv")
    df_gdp["Date"] = pd.to_datetime(df_gdp["Date"])

    # load housing data
    df_house = pd.read_csv("Datasets/average_house_price_quarterly.csv")
    df_house["Date"] = pd.to_datetime(df_house["Date"])
    df_house = df_house.sort_values("Date").reset_index(drop=True)

    # load Productivity (Nonfarm Business Labor Productivity)
    df_prod = pd.read_csv("Datasets/labor_productivity_quarterly.csv")
    #--------------------------------------------------------------------------

    # Filter for Nonfarm Business productivity rows
    df_prod = df_prod[
        (df_prod["Sector"] == "Nonfarm business sector") &
        (df_prod["Measure"] == "Labor productivity") &
        (df_prod["Units"].isin([
            "% Change same quarter 1 year ago",     # YoY
            "% Change from previous quarter"        # QoQ
        ]))
    ].copy()

    # Reshape from wide (columns per quarter) to long format (one row per quarter value)
    df_prod = df_prod.melt(
        id_vars=["Units"], 
        var_name="Quarter", 
        value_name="Value"
    )

    # Convert values to numeric (fix pivot errors)
    df_prod["Value"] = pd.to_numeric(df_prod["Value"], errors="coerce")

    #converting strings into year and quarters
    df_prod["Year"] = df_prod["Quarter"].str.extract(r"(^\d{4})")
    df_prod["Q"] = df_prod["Quarter"].str.extract(r"Q([1-4])")

    # Drop rows where quarter extraction failed
    df_prod = df_prod.dropna(subset=["Year", "Q"]).copy()

    #converting into real date
    df_prod["Date"] = pd.to_datetime(df_prod["Year"] + "-" + (df_prod["Q"].astype(int)*3 - 2).astype(str) + "-01")

    #keeping only the ones that are after 2013
    df_prod = df_prod[df_prod["Date"] >= "2013-01-01"].copy()


    # Pivot to wide form: one row per Date, columns YoY & QoQ
    df_prod_pivot = df_prod.pivot_table(
        index="Date",
        columns="Units",
        values="Value"
    ).reset_index()

    df_prod_pivot.columns = ["Date", "QoQ", "YoY"]  # rename for clarity

    # Sort by date
    df_prod_pivot = df_prod_pivot.sort_values("Date").reset_index(drop=True)

    # Final clean productivity df
    df_prod_final = df_prod_pivot.copy()
    # coerce - If a value can’t be converted to a number, turn it into NaN instead of crashing.
    df_prod_final["QoQ"] = pd.to_numeric(df_prod_final["QoQ"], errors="coerce")
    df_prod_final["YoY"] = pd.to_numeric(df_prod_final["YoY"], errors="coerce")

    #-------------------------------------------------------------------------------------------

    #create default layout for each tile

    #layout for cpi tile
    cpi_tile = Div(text="""
    <div style='width:150px; padding:10px; border:1px solid #ccc; border-radius:8px;'>CPI</div>""")

    #layout for unemployment tile
    unemp_tile = Div(text="""
    <div style="padding:10px;border:1px solid #ccc;border-radius:8px;
    width:180px;background:white;">
    <div style="font-size:11px;color:#666;">Unemployment</div>
    <div style="font-size:20px;font-weight:700;">--</div>
    <div style="font-size:12px;color:#444;">YoY: --</div>
    </div>
    """)

    #layout for gdp tile
    gdp_tile = Div(text="""
    <div style="padding:10px;border:1px solid #ccc;border-radius:8px;
    width:175px;background:white;">
    <div style="font-size:11px;color:#666;">GDP (YoY)</div>
    <div style="font-size:20px;font-weight:700;">--</div>
    <div style="font-size:12px;color:#444;">YoY on --</div>
    </div>
    """)

    house_tile = Div(text="""
    <div style="padding:10px;border:1px solid #ccc;border-radius:8px;
    width:175px;background:white;">
    <div style="font-size:11px;color:#666;">Avg House Price (USD)</div>
    <div style="font-size:20px;font-weight:700;">--</div>
    <div style="font-size:12px;color:#444;">YoY on --</div>
    </div>
    """)

    prod_tile = Div(text="""
    <div style="padding:10px;border:1px solid #ccc;border-radius:8px;
    width:175px;background:white;">
    <div style="font-size:11px;color:#666;">Productivity (Nonfarm)</div>
    <div style="font-size:20px;font-weight:700;">--</div>
    <div style="font-size:12px;color:#444;">YoY on --</div>
    </div>
    """)
    #---------------------------------------------------------------------

    #converts the date string into date type
    df["Date"] = pd.to_datetime(df["Date"])

    #create new index for each row
    df = df.sort_values("Date").reset_index(drop=True)

    #only have entries 2013 onwards
    df = df[df["Date"] >= "2013-01-02"].reset_index(drop=True)

    #yield curve time to maturity
    maturity = ["1 Mo","3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"]

    #data source for the plot
    src = ColumnDataSource(data=dict(m=[], y=[]))

    #yield curve figure
    plot = figure(height=260, width=620, title="Yield Curve", toolbar_location=None, x_range=maturity)
    plot.line("m", "y", source=src, line_width=2)
    plot.scatter("m", "y", source=src, size=6)
    #-------------------------------------------------------------------------------------------------
    #slider
    start_idx = df.index[df["Date"] == "2013-01-02"].tolist()[0] # set the start date to 02-01-2013
    date_slider = Slider(start=0, end=len(df)-1, value=start_idx, step=1, title="Date (index)")

    #helper function to get the closest cpi date
    def get_cpi_for_date(date):
        # find nearest CPI row on or before the selected date
        mask = cpi[cpi["Date"] <= date]
        if len(mask) == 0:
            return None
        return float(mask["CPIAUCSL_PC1"].iloc[-1])
    #--------------------------------------------------------------------------------------------------
    #callback function
    def on_slider_change(attr, old, new):
        i = int(new) # new input as user changes the slider
        row = df.iloc[i] # takes the input from the slider and finds the row in dataframe

        values = [float(row[m]) for m in maturity]  # yields for each maturity
        src.data = dict(m=maturity, y=values) # plot the list of maturities against the values

        # update the title of the plot with the date
        plot.title.text = f"Yield Curve — {row['Date'].strftime('%Y-%m-%d')}" 

        # data for cpi tile------------------------------------------------------------------------------------
        cpi_value = get_cpi_for_date(row["Date"])
        # update cpi tile
        if cpi_value is not None:
            cpi_tile.text = f"""
            <div style="padding:10px;border:1px solid #ccc;border-radius:8px;width:150px;background:white;">
                <div style="font-size:11px;color:#666;">CPI (YoY)</div>
                <div style="font-size:20px;font-weight:700;">{cpi_value:.2f}%</div>
                <div style="font-size:12px;color:#444;">On {row['Date'].strftime('%Y-%m-%d')}</div>
            </div>
            """
        else:
            cpi_tile.text = "No CPI Data"

        # data for unemployment------------------------------------------------------------------------------------
        # Filter unemployment data up to the selected slider date
        match = df_unemp[df_unemp["Date"] <= row["Date"]]

        if not match.empty:
            current_u = match.iloc[-1]["UNRATE"]

            # Get YoY unemployment: this month vs same month last year
            one_year_prior_mask = df_unemp[df_unemp["Date"] <= (row["Date"] - pd.DateOffset(years=1))]
            if not one_year_prior_mask.empty:
                # Value from 1 year earlier
                prev_u = one_year_prior_mask.iloc[-1]["UNRATE"]
                # YoY change = current minus last year's value
                yoy_change = current_u - prev_u
            else:
                # No previous year data available, so YoY is not computable
                yoy_change = np.nan

            # Format current unemployment value
            unemp_val = f"{current_u:.2f}%"
            
            # Format YoY change or show dash if unavailable
            if np.isnan(yoy_change):
                unemp_yoy = "-"
            else:
                sign = "+" if yoy_change >= 0 else ""
                unemp_yoy = f"{sign}{yoy_change:.2f} pp" # "pp" = percentage points
        else:
            # No unemployment data available up to that date
            unemp_val = "-"
            unemp_yoy = "-"

        # update unemployment tile
        unemp_tile.text = f"""
        <div style="padding:10px;border:1px solid #ccc;border-radius:8px;width:175px;background:white;">
        <div style="font-size:11px;color:#666;">Unemployment (current %)</div>
        <div style="font-size:20px;font-weight:700;">{unemp_val}</div>
        <div style="font-size:12px;color:#444;">{unemp_yoy} YoY on {row['Date'].strftime('%Y-%m-%d')}</div>
        </div>
        """

        # data for gdp tile------------------------------------------------------------------------------------
        match_gdp = df_gdp[df_gdp["Date"] <= row["Date"]]

        if not match_gdp.empty:
            current_gdp = match_gdp.iloc[-1]["GDP"]

            # same quarter last year (4 quarters earlier)
            prev_mask = df_gdp[df_gdp["Date"] <= (row["Date"] - pd.DateOffset(years=1))]
            if not prev_mask.empty:
                prev_gdp = prev_mask.iloc[-1]["GDP"]
                yoy_gdp = ((current_gdp - prev_gdp) / prev_gdp) * 100
            else:
                yoy_gdp = np.nan

            # Format GDP level in BILLIONS
            gdp_val = f"${current_gdp:,.0f}B"

            # Format YoY growth
            if np.isnan(yoy_gdp):
                gdp_yoy = "-"
            else:
                sign = "+" if yoy_gdp >= 0 else ""
                gdp_yoy = f"{sign}{yoy_gdp:.2f}%"
        else:
            gdp_val = "-"
            gdp_yoy = "-"
        
        #update gdp tile
        gdp_tile.text = f"""
        <div style="padding:10px;border:1px solid #ccc;border-radius:8px;width:175px;background:white;">
            <div style="font-size:11px;color:#666;">GDP (current, $ billions)</div>
            <div style="font-size:20px;font-weight:700;">{gdp_val}</div>
            <div style="font-size:12px;color:#444;">{gdp_yoy} YoY on {row['Date'].strftime('%Y-%m-%d')}</div>
        </div>
        """

        # data for housing tile------------------------------------------------------------------------------------
        match_house = df_house[df_house["Date"] <= row["Date"]]

        if not match_house.empty:
            current_house = match_house.iloc[-1]["ASPUS"]

            # value 1 year prior
            prev_mask = df_house[df_house["Date"] <= (row["Date"] - pd.DateOffset(years=1))]
            if not prev_mask.empty:
                prev_house = prev_mask.iloc[-1]["ASPUS"]
                yoy_house = ((current_house - prev_house) / prev_house) * 100
            else:
                yoy_house = np.nan

            # format dollars
            house_val = f"${current_house:,.0f}"

            # YoY formatting
            if np.isnan(yoy_house):
                house_yoy = "-"
            else:
                sign = "+" if yoy_house >= 0 else ""
                house_yoy = f"{sign}{yoy_house:.2f}%"
        else:
            house_val = "-"
            house_yoy = "-"

        # update housing tile
        house_tile.text = f"""
        <div style="padding:10px;border:1px solid #ccc;border-radius:8px;width:175px;background:white;">
        <div style="font-size:11px;color:#666;">Avg House Price (USD)</div>
        <div style="font-size:20px;font-weight:700;">{house_val}</div>
        <div style="font-size:12px;color:#444;">{house_yoy} YoY on {row['Date'].strftime('%Y-%m-%d')}</div>
        </div>
        """

        #data for productivity------------------------------------------------------------------------------------
        match_prod = df_prod_final[df_prod_final["Date"] <= row["Date"]]

        if not match_prod.empty:
            # latest YoY and QoQ values as of the slider date
            current_yoy = match_prod.iloc[-1]["YoY"]
            current_qoq = match_prod.iloc[-1]["QoQ"]

            # Format YoY productivity
            yoy_str = f"{current_yoy:.2f}%" if pd.notna(current_yoy) else "-"
            # Determine sign (+/-) for QoQ display
            qoq_sign = "+" if pd.notna(current_qoq) and current_qoq >= 0 else ""

            # Format QoQ productivity
            qoq_str = f"{qoq_sign}{current_qoq:.2f}%" if pd.notna(current_qoq) else "-"
        else:
            # If no productivity data yet for this date, show placeholders
            yoy_str = "-"
            qoq_str = "-"

        #update productivity tile
        prod_tile.text = f"""
        <div style="padding:10px;border:1px solid #ccc;border-radius:8px;width:200px;background:white;">
            <div style="font-size:11px;color:#666;">Labor Productivity (Nonfarm % YoY)</div>
            <div style="font-size:20px;font-weight:700;">{yoy_str}</div>
            <div style="font-size:11px;color:#555;">QoQ: {qoq_str} on {row['Date'].strftime('%Y-%m-%d')}</div>
        </div>
        """

    #call the callback function
    date_slider.on_change("value", on_slider_change)

    # trigger initial plot
    on_slider_change("value", None, date_slider.value)

    #organise the layout
    layout = column(
        Div(text="<h2>Yield Curve Viewer</h2>"),
        row(
            cpi_tile,
            unemp_tile,
            gdp_tile,
            house_tile,
            prod_tile
        ),
        plot,
        date_slider
    )
    doc.add_root(layout)

# create and start the server
server = Server({'/': modify_doc}, num_procs=1)
server.start()
print("Opening Bokeh application on http://localhost:5006/")
server.io_loop.add_callback(server.show, "/")
server.io_loop.start()