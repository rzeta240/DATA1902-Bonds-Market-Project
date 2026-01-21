import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from matplotlib import cm

from Scripts.dataset_reader import get_yield_curve_rates

yield_curve_rates = get_yield_curve_rates()

def curve_values(d): # Get the bond yield values for a specific date as a vector
    row = yield_curve_rates.loc[yield_curve_rates["Date"] == d] # All entries for a specific date

    if not row.empty: # If there is an entry for this date
        return row.iloc[:, 1:14].to_numpy().flatten() # Return a vector
    else:
        return np.array([]) # Otherwise none :(

def date(y, m, d):
    return datetime.date(datetime(y, m, d)) # Take some values and get back a datetime object

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# We slice out a subsection of the dataframe because using the whole thing looks horrible
dates = yield_curve_rates[yield_curve_rates["Date"] <= date(2025, 1, 1)] 
dates = dates[dates["Date"] >= date(2020, 1, 1)]["Date"]

# Convert the dates to the number of days since 2013, 1st Jan (for plotting)
dates.index = range( len( dates ) )
for i in range( len( dates ) ):
    dates[i] = dates[i].toordinal() - date(2013, 1, 1).toordinal()

# These are our X and Y axes
t = np.array(sorted(dates))
print(t)
m = np.array([1, 3, 6, 12, 24, 36, 60, 84, 120, 240, 360])

t, m = np.meshgrid(t, m)

y = t.copy()

# For each date, we find the curve values corresponding to it using curve_values
for i in range( len( t[0] ) ):
    d = datetime(2013,1,1) + timedelta(days=round(t[0][i]))
    d = datetime.date(d)

    c = curve_values(d)

    for j in range( len( t ) ): # Then update each point on the base of the graph with a Z value
        try:
            y[j][i] = c[j] * 100 # Convert to percentage
        except:
            y[j][i] = 0
            
print(y) # Checking

surf = ax.plot_surface(t, m, y, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False) # Actually plot it

plt.show()

# --- Below here is ChatGPT code, which I have no desire to understand because it is far outside the scope of this course and will not impact my success in any way
# --- Also, it is all commented, because this script is better used for plotting the yield surface in a little window on the computer, rather than printing out a 3D model
# --- We already printed it. You can look at it with this script. I kept the code for printing just for archival purposes.
# --- :)

# from stl import mesh

# ny, nx = t.shape
# vertices = np.column_stack([t.ravel(), m.ravel(), y.ravel()])

# faces = []
# for i in range(ny - 1):
#     for j in range(nx - 1):
#         # vertex indices for the 4 corners of the cell
#         v0 = i * nx + j
#         v1 = v0 + 1
#         v2 = v0 + nx
#         v3 = v2 + 1

#         # two triangles: (v0, v2, v1) and (v1, v2, v3)
#         faces.append([v0, v2, v1])
#         faces.append([v1, v2, v3])

# faces = np.array(faces)

# surface_data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
# surface_mesh = mesh.Mesh(surface_data)

# for i, f in enumerate(faces):
#     for j in range(3):
#         surface_mesh.vectors[i][j] = vertices[f[j], :]

# surface_mesh.save('surface.stl')