import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from Scripts.dataset_reader import get_yield_curve_rates

yield_curve_rates = get_yield_curve_rates()

def curve_values(d):
    row = yield_curve_rates.loc[yield_curve_rates["Date"] == d]

    if not row.empty:
        return row.iloc[:, 1:14].to_numpy().flatten()
    else:
        return np.array([])

def date(y, m, d):
    return datetime.date(datetime(y, m, d))

fig = plt.figure()
ax = fig.add_subplot(111)

fig.subplots_adjust(left=0.1, bottom=0.25)

m = np.array([1, 2, 3, 4, 6, 12, 24, 36, 60, 84, 120, 240, 360])
date_0 = date(2013, 1, 2)

[yield_curve] = ax.plot(m, curve_values(date_0), linewidth = 2, color="black")
ax.set_xlim([0,360])
ax.set_ylim([0, 10])

date_slider_ax = fig.add_axes([0.15, 0.05, 0.65, 0.03], facecolor="grey")
date_slider = Slider(date_slider_ax, "Date", 1, 4380, valinit = 1)

def on_slider_changed(val):
    d = datetime(2013,1,1,0,0) + timedelta(days=round(val - 1))
    d = datetime.date(d)

    c = curve_values(d)
    if len(c) > 1:
        yield_curve.set_ydata(c)
        fig.canvas.draw_idle()

date_slider.on_changed(on_slider_changed)

plt.show()