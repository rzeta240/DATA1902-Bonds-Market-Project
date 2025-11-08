import pandas as pd 
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import Select, CustomJS, ColumnDataSource, Paragraph, HoverTool
from bokeh.layouts import layout,column, row
from bokeh.server.server import Server
from bokeh.io import curdoc
from ridge_regression_redux import train_ridge_model

def modify_doc(doc):
    profit, mse, r2, dir_acc, y_true, y_pred = train_ridge_model(150, "6 Mo")
    x = list(range(len(profit)))
    cum_profit = np.cumsum(profit)
    #ensures equal length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    source = ColumnDataSource(data=dict(y_true=y_true, y_pred=y_pred))
    line_source = ColumnDataSource(data=dict(x=[max(min(y_true), min(y_pred)), min(max(y_true), max(y_pred))], y=[max(min(y_true), min(y_pred)), min(max(y_true), max(y_pred))]))
    profit_source = ColumnDataSource(data=dict(x=x, profit=profit, cum_profit=cum_profit))
    p = figure(width=550, height=400,title="Predicted vs Actuals (Look Forward Window: 150)", x_axis_label='Predicted Values', y_axis_label='Actual Values')
    points = p.scatter('y_pred', 'y_true', source=source, color="#043565")
    p.line('x', 'y', source=line_source, line_color="red", line_dash="dashed", line_width=2, legend_label="y=x")
    p.legend.label_text_font_size = "8pt"
    p1 = figure(width=550, height=400, title="Profit of Trading Strategy", x_axis_label='Month since implementation', y_axis_label='Profit ($)')
    bars = p1.vbar(x='x', top='profit', source=profit_source, color= "#5158BB", legend_label="Monthly Profit")
    line = p1.line('x', 'cum_profit', source=profit_source, color = "#EB4B98", line_width=2, legend_label="Cumulative Profit")
    p1.legend.label_text_font_size = "8pt"

    # --- Hover tool for bars ---
    hover_bars = HoverTool(
        renderers=[bars],
        tooltips=[
            ("Profit", "$@profit{0.00}")
        ]
    )

    # --- Hover tool for cumulative line ---
    hover_line = HoverTool(
        renderers=[line],
        tooltips=[
            ("Cumulative Profit", "$@cum_profit{0.00}")
        ]
    )
    # --- Hover tool for scatter ---
    hover_points = HoverTool(
        renderers=[points],
        tooltips=[
            ("Predicted", "@y_pred{0.00}"),
            ("Actual", "@y_true{0.00}")
        ]
    )
    p.add_tools(hover_points)
    p1.add_tools(hover_bars, hover_line)

    stats = Paragraph(text=f"""Model Performance Metrics:
    MSE: {mse:.4f} | R2: {r2:.4f} | Directional Accuracy: {dir_acc:.4f} | Total Profit: ${np.sum(profit):,.2f}
    """)

    look_fwd_window_select = Select(
        title="Look Forward Window",
        value="150",
        options=["3", "5", "7", "10", "15", "20", "25", "30", "40", "50", "60", "70", "80", "90", "100", "150", "200"]
    )

    structure_select = Select(
        title="Structure",
        value="6 Mo",
        options=["1 Mo", 
                 "3 Mo", 
                 "6 Mo", 
                 "1 Yr", 
                 "2 Yr", 
                 "3 Yr", 
                 "5 Yr", 
                 "10 Yr", 
                 "20 Yr", 
                 "30 Yr",
                "1 Mo_3 Mo_spread",
                "1 Mo_6 Mo_spread",
                "3 Mo_6 Mo_spread",
                "3 Mo_3 Yr_spread",
                "3 Mo_1 Yr_spread",
                "1 Yr_3 Yr_spread",
                "1 Yr_5 Yr_spread",
                "2 Yr_10 Yr_spread",
                "1 Yr_10 Yr_spread",
                "3 Yr_5 Yr_spread",
                "3 Yr_20 Yr_spread",
                "3 Yr_30 Yr_spread",
                "5 Yr_10 Yr_spread",
                "5 Yr_20 Yr_spread",
                "5 Yr_30 Yr_spread",
                "10 Yr_30 Yr_spread",
                "10 Yr_20 Yr_spread"
            ]
    )

    def update_plot(attr, old, new):
        look_fwd_window = int(look_fwd_window_select.value)
        structure = structure_select.value
        profit, mse, r2, dir_acc, y_true, y_pred = train_ridge_model(look_fwd_window, structure)
        #ensures equal length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        source.data = dict(y_true=y_true, y_pred=y_pred)
        p.title.text = f"Predicted vs Actuals (Look Forward Window: {look_fwd_window})"
        line_source.data = dict(x=[max(min(y_true), min(y_pred)), min(max(y_true), max(y_pred))], y=[max(min(y_true), min(y_pred)), min(max(y_true), max(y_pred))])
        profit_source.data = dict(x=list(range(len(profit))), profit=profit, cum_profit=np.cumsum(profit))
        p1.title.text = f"Profit of Trading Strategy (Look Forward Window: {look_fwd_window})"
        stats.text = f"""Model Performance Metrics:
        MSE: {mse:.4f} | R2: {r2:.4f} | Directional Accuracy: {dir_acc:.4f} | Total Profit: ${np.sum(profit):,.2f}
        """

    look_fwd_window_select.on_change("value", update_plot)
    structure_select.on_change("value", update_plot)

    doc.add_root(column(row(look_fwd_window_select, structure_select), row(p, p1), stats))

# Create and start server
server = Server({'/': modify_doc}, num_procs=1)
server.start()

print("Opening Bokeh application on http://localhost:5006/")
server.io_loop.add_callback(server.show, "/")
server.io_loop.start()
