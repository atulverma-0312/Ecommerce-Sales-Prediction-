from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"

df = pd.read_csv(DATA_DIR / "cleaned_sales.csv")

df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Aggregate transaction-level data into daily forecasting features
daily = (
    df.groupby("Order_Date")
    .agg(
        Sales=("Sales", "sum"),
        Orders=("Order_ID", "nunique"),
        Units_Sold=("Quantity", "sum"),
    )
    .sort_index()
)

# Include dates without orders as zero sales
daily = daily.asfreq("D", fill_value=0)

data = daily.copy()

# Date features
data["Year"] = data.index.year
data["Quarter"] = data.index.quarter
data["Month"] = data.index.month
data["Day"] = data.index.day
data["DayOfWeek"] = data.index.dayofweek
data["Weekend"] = (data.index.dayofweek >= 5).astype(int)
data["DayOfYear"] = data.index.dayofyear
data["IsMonthStart"] = data.index.is_month_start.astype(int)
data["IsMonthEnd"] = data.index.is_month_end.astype(int)

# Cyclical date features preserve the wraparound between weeks and years
data["Weekday_Sin"] = np.sin(data["DayOfWeek"] * 2 * np.pi / 7)
data["Weekday_Cos"] = np.cos(data["DayOfWeek"] * 2 * np.pi / 7)
data["YearDay_Sin"] = np.sin(data["DayOfYear"] * 2 * np.pi / 365.25)
data["YearDay_Cos"] = np.cos(data["DayOfYear"] * 2 * np.pi / 365.25)

# Historical sales features
data["Lag_1"] = data["Sales"].shift(1)
data["Lag_7"] = data["Sales"].shift(7)
data["Lag_14"] = data["Sales"].shift(14)
data["Lag_28"] = data["Sales"].shift(28)
data["Lag_30"] = data["Sales"].shift(30)

data["Rolling_7"] = data["Sales"].shift(1).rolling(7).mean()
data["Rolling_14"] = data["Sales"].shift(1).rolling(14).mean()
data["Rolling_28"] = data["Sales"].shift(1).rolling(28).mean()
data["Sales_Std_7"] = data["Sales"].shift(1).rolling(7).std()
data["Sales_Std_28"] = data["Sales"].shift(1).rolling(28).std()

# Past order volume and units sold can help explain sales fluctuations
for column in ("Orders", "Units_Sold"):
    data[f"{column}_Lag_1"] = data[column].shift(1)
    data[f"{column}_Lag_7"] = data[column].shift(7)
    data[f"{column}_Rolling_7"] = data[column].shift(1).rolling(7).mean()

# Keep same-day order metrics out of the output to avoid target leakage
data = data.drop(columns=["Orders", "Units_Sold"])

# Remove rows with missing lag values
data = data.dropna()

data.index.name = "Order_Date"
data.to_csv(DATA_DIR / "feature_engineered.csv")

print(data.head())
print("Feature Engineering Completed!")