import numpy as np
import pandas as pd

MONTH_MAP = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
             7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}

NUM_COLS = ["DayOfWeek", "Promo", "SchoolHoliday", "Promo2", "Month", "Day", "WeekOfYear",
            "CompetitionDistance", "CompetitionMonthsOpen", "Promo2WeeksOpen",
            "IsPromo2Month", "IsWeekend"]
CAT_COLS = ["StoreType", "Assortment", "StateHoliday"]
FEATURE_COLS = NUM_COLS + CAT_COLS + ["Store"]


def build_features(store_row, dates, promo=0, school_holiday=0, state_holiday="0"):
    """Build the model input for one store and a list of dates.

    promo / school_holiday / state_holiday can be a single value or one value per date.
    Promotions never run on weekends in the training data, so Promo is forced to 0 there.
    """
    d = pd.DataFrame({"Date": pd.to_datetime(list(dates))})
    d["Store"] = int(store_row["Store"])
    d["DayOfWeek"] = d["Date"].dt.dayofweek + 1
    d["Year"] = d["Date"].dt.year
    d["Month"] = d["Date"].dt.month
    d["Day"] = d["Date"].dt.day
    d["WeekOfYear"] = d["Date"].dt.isocalendar().week.astype(int)

    d["Promo"] = np.where(d["DayOfWeek"] <= 5, promo, 0).astype(int)
    d["SchoolHoliday"] = school_holiday
    d["StateHoliday"] = state_holiday
    d["StateHoliday"] = d["StateHoliday"].astype(str)

    for c in ["StoreType", "Assortment", "CompetitionDistance", "Promo2"]:
        d[c] = store_row[c]

    # same cleaning as the notebook: implausible competition years -> missing
    comp_year = store_row["CompetitionOpenSinceYear"]
    if pd.notna(comp_year) and comp_year < 1990:
        comp_year = np.nan
    comp_month = store_row["CompetitionOpenSinceMonth"]

    d["CompetitionMonthsOpen"] = ((d["Year"] - comp_year) * 12 + (d["Month"] - comp_month)
                                  ).clip(lower=0).fillna(0)
    d["Promo2WeeksOpen"] = ((d["Year"] - store_row["Promo2SinceYear"]) * 52
                            + (d["WeekOfYear"] - store_row["Promo2SinceWeek"])
                            ).clip(lower=0).fillna(0)

    interval = store_row["PromoInterval"]
    d["IsPromo2Month"] = [int(isinstance(interval, str) and MONTH_MAP[m] in interval.split(","))
                          for m in d["Month"]]
    d["IsWeekend"] = (d["DayOfWeek"] >= 6).astype(int)
    return d[FEATURE_COLS]
