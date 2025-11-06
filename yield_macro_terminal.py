# To run this file use the following command in the terminal
#bokeh serve --show yield_macro_terminal.py
import math
from pathlib import Path
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Div, Select, Slider, NumeralTickFormatter
from bokeh.plotting import figure

#import from ridge regression
from ridge_regression_redux import train_ridge_model


BASE_DIR = Path(__file__).parent.resolve()

# File helpers
def find_file(fname: str) -> Path | None:
    """Try ./fname then ./Datasets/fname; return absolute Path or None."""
    here = BASE_DIR / fname
    ds = BASE_DIR / "Datasets" / fname
    if here.exists(): return here.resolve()
    if ds.exists():   return ds.resolve()
    return None

def load_csv(fname: str) -> tuple[pd.DataFrame, str]:
    """
    Load CSV from ./ or ./Datasets; return (df, debug_str).
    df has Date parsed if present. On error, returns empty df with Date col.
    """
    p = find_file(fname)
    if p is None:
        msg = f"[WARN] Missing file: {fname} (looked in {BASE_DIR} and {BASE_DIR/'Datasets'})"
        print(msg)
        return pd.DataFrame({"Date": []}), msg
    try:
        df = pd.read_csv(p)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        msg = f"[OK] Loaded {fname} from: {p}"
        print(msg)
        return df, msg
    except Exception as e:
        msg = f"[WARN] Failed to read {fname} at {p}: {e}"
        print(msg)
        return pd.DataFrame({"Date": []}), msg

def pick_numeric_column(df: pd.DataFrame, prefer: list[str] | None = None) -> str | None:
    """
    Return a numeric column name. If 'prefer' provided, try those first (coerce).
    """
    if df.empty:
        return None
    # Try preferred first
    if prefer:
        for c in prefer:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().any():
                    df[c] = s
                    return c
    # Otherwise, the first numeric-ish after Date
    for c in df.columns:
        if c == "Date": 
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            df[c] = s
            return c
    return None

def last_two_values(series: pd.Series) -> tuple[float, float] | tuple[float, float]:
    """Return (curr, prev) last 2 finite values; if not available, returns (nan, nan)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return math.nan, math.nan
    return float(s.iloc[-1]), float(s.iloc[-2])


# Load datasets
yc_df, yc_dbg   = load_csv("yield_curve_rates_daily.csv")
unemp, un_dbg   = load_csv("unemployment_rate_monthly.csv")
gdp, gdp_dbg    = load_csv("GDP_quarterly.csv")
house, house_dbg= load_csv("average_house_price_quarterly.csv")
prod, prod_dbg  = load_csv("labor_productivity_quarterly.csv")

if not yc_df.empty and "Date" in yc_df.columns:
    yc_df = yc_df.sort_values("Date")

debug_lines = [yc_dbg, un_dbg, gdp_dbg, house_dbg, prod_dbg]

#CPI tile (hard-prefer CPIAUCSL_PC1)
def cpi_tile():
    p = find_file("consumer_price_index_quarterly.csv")
    if p is None:
        msg = f"[WARN] CPI file not found (looked in {BASE_DIR} and {BASE_DIR/'Datasets'})"
        print(msg)
        return Div(text=f"<b style='color:red;'>CPI Error:</b> file not found"), msg

    try:
        df = pd.read_csv(p)
        # Parse date if present
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Prefer CPIAUCSL_PC1; otherwise find first numeric col
        col = pick_numeric_column(df, prefer=["CPIAUCSL_PC1"])
        if col is None:
            msg = f"[WARN] CPI: no numeric column found in {p.name}"
            print(msg)
            return Div(text=f"<b style='color:red;'>CPI Error:</b> no numeric column"), msg

        curr, prev = last_two_values(df[col])
        if not (np.isfinite(curr) and np.isfinite(prev)):
            msg = f"[WARN] CPI: insufficient numeric values in column '{col}'"
            print(msg)
            return Div(text=f"<b style='color:red;'>CPI Error:</b> insufficient data"), msg

        delta = curr - prev
        sign = "+" if delta >= 0 else ""

        # format as percent (assumes values are already percent like 1.23)
        def fmt(v): return f"{v:.2f}%"

        tile_html = f"""
        <div style="padding:14px;border:1px solid #ccc;border-radius:10px;
                    width:200px;background:white;">
          <div style="font-size:11px;color:#777;">CPI (latest)</div>
          <div style="font-size:22px;font-weight:700;color:#111;">{fmt(curr)}</div>
          <div style="font-size:12px;color:#555;">{sign}{fmt(delta)} vs prev</div>
        </div>
        """
        msg = f"[OK] CPI from {p} using column '{col}' (curr={curr:.4f}, prev={prev:.4f})"
        print(msg)
        return Div(text=tile_html), msg

    except Exception as e:
        msg = f"[WARN] CPI read error at {p}: {e}"
        print(msg)
        return Div(text=f"<b style='color:red;'>CPI Error:</b> {e}"), msg

cpi_div, cpi_dbg = cpi_tile()
debug_lines.append(cpi_dbg)

# Productivity tile
def prod_tile_values(df: pd.DataFrame) -> tuple[float, float, str]:
    col = pick_numeric_column(df)
    if col is None:
        return math.nan, math.nan, "(none)"
    curr, prev = last_two_values(df[col])
    return curr, (curr - prev), col

prod_v, prod_d, prod_col = prod_tile_values(prod)

# Shared tile helpers
def fmt_pct(x):   return "—" if not np.isfinite(x) else f"{x:.1f}%"
def fmt_money(x): return "—" if not np.isfinite(x) else f"${x:,.0f}"

def last_and_delta(df):
    if df.empty or "Date" not in df.columns or len(df) < 2:
        return math.nan, math.nan, "(none)"
    col = pick_numeric_column(df)
    if col is None:
        return math.nan, math.nan, "(none)"
    curr, prev = last_two_values(df[col])
    if not (np.isfinite(curr) and np.isfinite(prev)):
        return math.nan, math.nan, col
    return curr, curr - prev, col

def tile(label, val, delta, fmt):
    sign = "+" if np.isfinite(delta) and delta >= 0 else ""
    return Div(text=f"""
    <div style="padding:14px;border:1px solid #ccc;border-radius:10px;
                width:200px;background:white;">
      <div style="font-size:11px;color:#777;">{label}</div>
      <div style="font-size:22px;font-weight:700;color:#111;">{fmt(val)}</div>
      <div style="font-size:12px;color:#555;">{sign}{fmt(delta)} vs prev</div>
    </div>
    """)

unemp_v, unemp_d, un_col = last_and_delta(unemp)
gdp_v,   gdp_d,   gdp_col = last_and_delta(gdp)
house_v, house_d, house_col = last_and_delta(house)

# Yield curve plot
DEFAULT_TENORS = ["1 Mo","3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"]
yc_src = ColumnDataSource(dict(x=[], y=[], t=[]))

p_yc = figure(height=250, width=600, title="Yield Curve", toolbar_location=None,
              x_range=DEFAULT_TENORS)
p_yc.line(x="t", y="y", source=yc_src, line_width=2)
p_yc.scatter(x="t", y="y", source=yc_src, size=6)
p_yc.yaxis.formatter = NumeralTickFormatter(format="0.00")

def norm(v): 
    try:
        v = float(v)
    except:
        return math.nan
    return v*100 if v <= 1.2 else v

def update_curve(idx):
    if yc_df.empty:
        return
    row = yc_df.iloc[idx]
    ten, y = [], []
    for t in DEFAULT_TENORS:
        if t in yc_df.columns:
            ten.append(t)
            y.append(norm(row[t]))
    yc_src.data = dict(x=list(range(len(ten))), y=y, t=ten)
    d = row["Date"]
    p_yc.title.text = f"Yield Curve — {d.strftime('%Y-%m-%d') if pd.notnull(d) else '—'}"

# Slider
date_slider = Slider(
    start=0, 
    end=len(yc_df)-1,
    value=len(yc_df)-1,
    step=1,
    title="Yield Curve Date (index)"
)

# Date label under slider (real date text)
date_label = Div(text="", styles={"font-size": "12px", "color": "#444"})

def on_date_change(attr, old, new):
    idx = int(new)
    update_curve(idx)
    d = yc_df.iloc[idx]["Date"]
    date_label.text = f"<b>Date:</b> {d.strftime('%Y-%m-%d')}"

date_slider.on_change("value", on_date_change)

#Model Controls
WINDOW_OPTIONS = ["3","5","7","10","15","20","25","30","40","50","60","70","80","90","100","150","200"]
STRUCTURE_OPTIONS = [
"1 Mo","3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","10 Yr","20 Yr","30 Yr",
"1 Mo_3 Mo_spread","1 Mo_6 Mo_spread","3 Mo_6 Mo_spread","3 Mo_3 Yr_spread",
"3 Mo_1 Yr_spread","1 Yr_3 Yr_spread","1 Yr_5 Yr_spread","2 Yr_10 Yr_spread",
"1 Yr_10 Yr_spread","3 Yr_5 Yr_spread","3 Yr_20 Yr_spread","3 Yr_30 Yr_spread",
"5 Yr_10 Yr_spread","5 Yr_20 Yr_spread","5 Yr_30 Yr_spread","10 Yr_30 Yr_spread","10 Yr_20 Yr_spread"
]

spread_sel = Select(title="Structure", value="6 Mo", options=STRUCTURE_OPTIONS)
win_sel    = Select(title="Lookahead (days)", value="150", options=WINDOW_OPTIONS)

signal_div = Div(text="<b>Signal:</b> —")
stats_div  = Div(text="")

def update_model():
    try:
        w = int(win_sel.value)
        s = spread_sel.value
        profit, mse, r2, acc, y_true, y_pred = train_ridge_model(w, s)
        last = y_pred[-1] if len(y_pred) else float("nan")

        if np.isfinite(last) and last > 0:
            signal_div.text = "<b style='color:#2e7d32;'>Signal: LONG</b>"
        elif np.isfinite(last) and last < 0:
            signal_div.text = "<b style='color:#c62828;'>Signal: SHORT</b>"
        else:
            signal_div.text = "<b>Signal:</b> —"

        stats_div.text = f"R²: {r2:.3f} | MSE: {mse:.4f} | DirAcc: {acc*100:.1f}% | Profit: ${np.sum(profit):,.0f}"
    except Exception as e:
        stats_div.text = f"Model error: {e}"

spread_sel.on_change("value", lambda a,o,n: update_model())
win_sel.on_change("value", lambda a,o,n: update_model())

#Layout
header = Div(text=f"""
<h2 style="margin:0;">Yield & Macro Strategy Terminal</h2>
<div style="font-size:12px;color:#777;">Macro Dashboard · Yield Curve</div>
""")

tiles = row(
    cpi_div,
    tile("Unemployment",   unemp_v,  unemp_d,  lambda v: "—" if not np.isfinite(v) else f"{v:.2f}%"),
    tile("GDP (bn)",       gdp_v,    gdp_d,    lambda v: "—" if not np.isfinite(v) else f"${v:,.0f}"),
    tile("Productivity",   prod_v,   prod_d,   lambda v: "—" if not np.isfinite(v) else f"{v:.2f}%"),
    tile("House Price",    house_v,  house_d,  lambda v: "—" if not np.isfinite(v) else f"${v:,.0f}"),
)

controls = column(
    Div(text="<b>Model Controls</b>"),
    spread_sel,
    win_sel,
    Div(text="<b>Signal</b>"),
    signal_div,
    Div(text="<b>Stats</b>"),
    stats_div,
)

layout = column(
    header,
    tiles,
    row(p_yc, column(row(date_slider, date_label), controls)),
    sizing_mode="stretch_width"
)

update_curve(int(date_slider.value))
update_model()

curdoc().add_root(layout)
curdoc().title = "Yield Dashboard"