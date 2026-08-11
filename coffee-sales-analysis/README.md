# Coffee Sales Analysis & Forecasting

Time series analysis and forecasting of daily coffee sales revenue from a vending-machine transaction dataset, comparing Facebook Prophet against SARIMA baselines.

## Dataset

Raw transaction-level data (`index_1.csv`) with columns including `date`, `datetime`, `card`, `cash_type`, `coffee_name`, and `money`. Each row is a single sale; `card` and `cash_type` are dropped early as they add no predictive value at this stage.

## Approach

1. **EDA** — average sales per coffee type, transaction volume per day, and a check on whether per-category daily volume is high enough to model individually. It isn't (too sparse), so the analysis aggregates to **total daily sales revenue** and models that as a single series.
2. **Decomposition** — `seasonal_decompose` (multiplicative, weekly period) on daily revenue. Residuals show fanning-out over time, suggesting heteroscedasticity (growing variance) rather than clean seasonality.
3. **Feature engineering** — lag features (`lag_1`, `lag_2`, `lag_3`) and a US-holiday calendar (used as a location proxy, since the dataset has no location field), for use as Prophet regressors/holidays.
4. **Prophet model** — grid search over seasonality mode, seasonality/holiday prior scales, and changepoint prior scale, validated via Prophet's built-in cross-validation. Fit with the best-found parameters and evaluated on a held-out 30-day test window.
5. **SARIMA baselines** — since Prophet underperformed on the test window, two SARIMAX models are fit as a simpler alternative:
   - Raw-scale SARIMA (order tuned via a custom rolling-origin CV objective using `mango`'s tuner).
   - Log-transformed SARIMA, to test whether stabilizing variance improves the fit.
6. **Diagnostics** — stationarity check via ADF test, and a rolling mean/std plot across the train/test split to check for a structural break between the two periods.

## Results

| Model | MAPE | MAE |
|---|---|---|
| Prophet (tuned) | — poor fit on test window, plotted but not the working model | — |
| SARIMA (raw scale) | printed at runtime | printed at runtime |
| SARIMA (log-transformed) | printed at runtime | printed at runtime |

**Findings:**
- Prophet did poorly on the test data, likely due to a structural break between train and test periods (confirmed by the rolling mean/std plot) and/or overfitting given the limited training data (~350 daily observations).
- The log-transformed SARIMA performed *worse* than the raw-scale version across every order tried, contrary to the initial hypothesis that stabilizing variance would help — kept in the script as documented exploratory work.
- The raw-scale SARIMA is the current working baseline.

## Requirements

Beyond the repo-level `requirements.txt` (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`), this script also needs:

```
statsmodels
prophet
holidays
mango-py
```

## Running

The script currently reads the dataset from a hardcoded absolute path (`index_1.csv`), so update that path locally before running:

```bash
python coffee_sales.py
```

The SARIMAX hyperparameter search (via `mango`) is commented out by default since it's slow — the script uses the best parameters already found from a prior run. Uncomment that block to re-run the search.
