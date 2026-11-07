import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))
from features import build_features  # noqa: E402

st.set_page_config(page_title="Store Sales Forecast", page_icon="🛒", layout="wide")

LAST_DATA_DAY = pd.Timestamp("2015-07-31")
TYPICAL_ERROR = 0.12   # test RMSPE ≈ 0.123


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")


@st.cache_data
def load_store_info():
    return pd.read_csv(APP_DIR / "store_info.csv").set_index("Store", drop=False)


@st.cache_data
def load_open_rates():
    return pd.read_csv(APP_DIR / "store_open_rates.csv")


model = load_model()
stores = load_store_info()
rates = load_open_rates()

st.title("🛒 Store Sales Forecast")
st.write("Pick a store and a period to see the forecasted daily sales, and compare "
         "running promotions on weekdays with not running them.")

# ---------------- inputs ----------------
with st.sidebar:
    st.header("Scenario")
    store_id = st.selectbox("Store", stores.index.tolist())
    start = st.date_input("First day", value=pd.Timestamp("2015-08-03").date(),
                          min_value=pd.Timestamp("2015-08-01").date(),
                          max_value=pd.Timestamp("2016-12-31").date())
    days = st.slider("Number of days", 7, 42, 28)
    promo_on = st.toggle("Run promotions on weekdays", value=True)
    st.caption("Promotions never run on Saturday or Sunday in the historical data, "
               "so weekends are always forecast without a promotion.")

row = stores.loc[store_id]
dates = pd.date_range(pd.Timestamp(start), periods=days)

if dates[-1] > LAST_DATA_DAY + pd.Timedelta(days=42):
    st.warning("The model was built for a 6-week horizon after the last data (31 Jul 2015). "
               "Forecasts further out are less reliable.")

# ---------------- forecast ----------------
open_rate = rates[rates["Store"] == store_id].set_index("DayOfWeek")["open_rate"]
usually_closed = np.array([open_rate.get(d, 1.0) < 0.5 for d in (dates.dayofweek + 1)])


def forecast(promo):
    pred = np.expm1(model.predict(build_features(row, dates, promo=promo)))
    return np.where(usually_closed, 0.0, pred)


base = forecast(0)
plan = forecast(1) if promo_on else base

# ---------------- results ----------------
c1, c2, c3 = st.columns(3)
c1.metric("Forecast total sales", f"{plan.sum():,.0f}")
c2.metric("Open days in period", int((~usually_closed).sum()))
if promo_on and base.sum() > 0:
    gain = plan.sum() - base.sum()
    c3.metric("Gain vs. no promotion", f"{gain:,.0f}", f"{gain / base.sum() * 100:+.1f}%")
else:
    c3.metric("Gain vs. no promotion", "-")

chart_df = pd.DataFrame({"No promotion": base}, index=dates)
if promo_on:
    chart_df["Weekday promotions"] = plan
st.subheader(f"Daily forecast, store {store_id}")
st.line_chart(chart_df)

with st.expander("Daily numbers"):
    note = np.where(usually_closed, "usually closed, forecast 0",
                    np.where(promo_on & (dates.dayofweek < 5), "promo day", ""))
    st.dataframe(pd.DataFrame({"Date": dates.date, "Day": dates.day_name(),
                               "Forecast": plan.round(0), "Note": note}),
                 hide_index=True, width="stretch")

with st.expander("About this store"):
    dist = row["CompetitionDistance"]
    st.write(f"Store type **{row['StoreType']}**, assortment **{row['Assortment']}**, "
             f"nearest competitor: {'unknown' if pd.isna(dist) else f'{dist:,.0f} m'}.")

st.info(f"Typical error is about ±{TYPICAL_ERROR:.0%} per store-day (RMSPE 0.123 on a held-out "
        "6-week test). The promotion gain is what the model attributes to promotions, "
        "not a proven causal effect. Holidays were not evaluated, and stores with unusual "
        "jumps in sales can be under-predicted.")
st.caption("Data: Rossmann Store Sales (Kaggle). Model: XGBoost trained on 2013 – mid-2015.")
