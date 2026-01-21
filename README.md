# US Treasury Yield Curve Prediction from Macroeconomic Indicators

## Overview  
This group coursework project investigates whether current macroeconomic conditions (inflation, unemployment, GDP, housing prices, and labor productivity) can be used to predict future changes in the U.S. Treasury bond yield curve.  

Using historical government data (2013–2024), we engineered economic features and train **statistical and machine learning models** to forecast both the direction and magnitude of yield curve movements, with an emphasis on the widely traded 10Y–2Y yield spread.

---

## Data Sources

| Dataset | Source | Frequency | Time Range |
|---------|--------|-----------|------------|
| Treasury Yield Curve | [U.S. Department of the Treasury](https://home.treasury.gov/interest-rates-data-csv-archive) | Daily | 2013–2024 |
| CPI (Inflation) | [Bureau of Labor Statistics (FRED)](https://fred.stlouisfed.org/series/CPIAUCSL) | Monthly / Quarterly | 2013–2024 |
| Unemployment Rate | [Bureau of Labor Statistics (FRED)](https://fred.stlouisfed.org/series/UNRATE) | Monthly / Quarterly | 2013–2024 |
| GDP | [Bureau of Economic Analysis (FRED)](https://fred.stlouisfed.org/series/GDP) | Quarterly | 2013–2024 |
| Housing Prices (ASPUS) | [U.S. Census Bureau / HUD (FRED)](https://fred.stlouisfed.org/series/ASPUS) | Quarterly | 2013–2024 |
| Labor Productivity | [Bureau of Labor Statistics](https://www.bls.gov/productivity/tables/) | Quarterly | 2013–2024 |

---

## Feature Engineering  
We transformed raw macroeconomic and yield data into **model-ready features**, including:

- **Yield curve spreads** (e.g., 10Y–2Y, short/long horizon structures)  
- **Quarterly aggregation of daily bond yields**  
- **Exponential smoothing** of macro indicators to reduce noise  
- **Lagged economic features** to capture delayed market response  
- **Directional labels** for classification models (positive vs. negative yield changes)  

---

## Modeling Approach  

We trained separate models for multiple yield structures and forecast horizons, allowing flexible trading and forecasting use cases.

### Ridge Regression  
**Purpose:** Predicts magnitude and direction of yield changes  
- Handles multicollinearity across macro indicators  
- Regularization tuned via validation R² optimization  
- Best suited for risk-balanced trading strategies

### Random Forest Regression  
**Purpose:** Captures non-linear macro-financial relationships
- Ensemble learning for regime shifts and interaction effects  
- Hyperparameter tuning via grid search  
- Strong performance in volatile market periods

### Logistic Regression  
**Purpose:** Predicts **direction only (up/down)**  
- Converts yield changes into binary trading signals  
- Optimised for accuracy and F1 score

---

## Evaluation Metrics  

In addition to standard ML metrics, we introduced **finance-specific performance measures**:

### Statistical Metrics  
- R² — Variance explained  
- SE — Prediction error magnitude  
- Accuracy / F1 Score — Directional classification quality  

### Trading Metrics  
- Directional Accuracy (Hit Rate) — Correct buy/sell signals  
- Simulated Profit — Trading performance based on predicted yield direction and realized spread changes  

---

## Key Findings  

- Labor productivity and unemployment show moderate inverse correlations with **short- to medium-term yields**  
- Housing price growth tends to align with yield curve steepening  
- GDP growth shows weaker direct predictive power  
- Non-linear models (Random Forests) outperform linear models during **high-volatility economic regimes**  
- Directional models provide reliable signal confirmation for trading strategies

---

## Visualisations  

The project includes several exploratory and analytical visuals:

- 3D Yield Surface — Term structure evolution over time  
- Macroeconomic vs Yield Spread Comparisons**  
  - CPI vs 10Y–2Y spread  
  - Housing prices vs yield slope  
  - Unemployment vs yield inversion  
- Labor Productivity Heatmap — Sectoral volatility and yield correlation

---

## Tools/skills used
- **Python**  
- **pandas, NumPy** — Data processing  
- **scikit-learn** — Modeling & evaluation  
- **Matplotlib / Seaborn** — Visualisation  
- **Git** — Version control  

- Time-series feature engineering  
- Financial data modelling  
- Macroeconomic analysis  
- Machine learning model selection and tuning  
- Quantitative trading evaluation  
- Data cleaning and transformation at scale  
- Visualisation for economic interpretation  

