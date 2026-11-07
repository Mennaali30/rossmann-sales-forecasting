# Store Sales Forecasting for a Retail Chain (Rossmann)

A retail chain with 1,115 stores must plan staff and stock six weeks ahead. This project forecasts daily sales per store, and lets planners compare running weekday promotions against not running them.

## Results (untouched test set, 2015-06-20 → 2015-07-31)

| Model | RMSPE | MAE | RMSE | R² |
|---|---|---|---|---|
| Baseline: store mean | 0.3041 | 1,441 | 1,955 | 0.591 |
| **XGBoost (tuned)** | **0.1231** | **635** | **925** | **0.908** |

The percentage error drops from 30.4% to 12.3% compared with a per-store average. Models compared on a validation set: Ridge (0.214), Random Forest (0.142), HistGradientBoosting (0.138 tuned) and XGBoost (0.136 tuned).

## What I did
- Cleaned ~1M daily records (closed days, zero-sales days, implausible competition dates).
- Used a **time-based** train / validation / test split and time-series cross-validation (no random split, no leakage).
- Built features (competition age, Promo2 duration, Promo2 renewal months, weekend) and a store-level mean encoder, all inside a scikit-learn Pipeline.
- Tuned XGBoost and HistGradientBoosting with randomized search; selected XGBoost for the best validation score and a 12 MB model size.
- Saved the full pipeline with `joblib` and deployed a Streamlit app: store, dates and promotion toggle in, forecast chart and gain versus no promotion out.

## Key findings
- Promotions are the strongest driver: on weekdays the model implies about +33% sales (raw averages: +42%), largest on Monday (+46%) and smallest on Friday (+23%).
- Promotions never run on weekends in the data, so weekend promotion effects cannot be estimated and the app forces them off.
- Typical error is about ±12% per store-day. A few stores (for example 292, 909, 876) are under-predicted; 29 of 1,115 stores have RMSPE above 0.20.

## Limitations
One 6-week test window with no state holidays; no December-peak evaluation; no lag, weather or local-event features; the promo effect is an association, not proven causation.

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
To re-run the notebook, download `train.csv` and `store.csv` from the [Kaggle competition](https://www.kaggle.com/c/rossmann-store-sales) (accept the rules first) and set `DATA_PATH` in the first cell.

## Data
[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales), Kaggle (~1M daily records, 1,115 stores, 2013–2015). Not included in this repo.

## Tools
Python, pandas, scikit-learn, XGBoost, Streamlit, joblib.

Author: Menna Ali Abdelrahman · NextGen Institute Machine Learning final project
 **Live app:** [https://rossmann-forecast-menna.streamlit.app/]
