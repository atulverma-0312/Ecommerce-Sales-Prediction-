from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"

# Load the saved model and the exact feature order used during training.
model = joblib.load(MODEL_DIR / "sales_model.pkl")
features = joblib.load(MODEL_DIR / "features.pkl")

# Build daily sales, order-count, and units-sold history from the cleaned data.
transactions = pd.read_csv(DATA_DIR / "cleaned_sales.csv", parse_dates=["Order_Date"])
daily = (
    transactions.groupby("Order_Date")
    .agg(
        Sales=("Sales", "sum"),
        Orders=("Order_ID", "nunique"),
        Units_Sold=("Quantity", "sum"),
    )
    .sort_index()
    .asfreq("D", fill_value=0)
)

sales_history = daily["Sales"].tolist()
orders_history = daily["Orders"].tolist()
units_history = daily["Units_Sold"].tolist()

if len(sales_history) < 30:
    raise ValueError("At least 30 days of history are required to create the model features.")

# Forecast one year ahead. Each forecast is used as history for the next day.
future_dates = pd.date_range(
    start=daily.index[-1] + pd.Timedelta(days=1),
    periods=365,
    freq="D",
)
forecast_rows = []

for date in future_dates:
    # Future order counts and units are unknown, so estimate them from the
    # same weekday one week earlier, just like a simple seasonal baseline.
    estimated_orders = orders_history[-7]
    estimated_units = units_history[-7]
    day_of_week = date.day_of_week
    day_of_year = date.day_of_year

    row = {
        "Year": date.year,
        "Quarter": date.quarter,
        "Month": date.month,
        "Day": date.day,
        "DayOfWeek": day_of_week,
        "Weekend": int(day_of_week >= 5),
        "DayOfYear": day_of_year,
        "IsMonthStart": int(date.is_month_start),
        "IsMonthEnd": int(date.is_month_end),
        "Weekday_Sin": np.sin(day_of_week * 2 * np.pi / 7),
        "Weekday_Cos": np.cos(day_of_week * 2 * np.pi / 7),
        "YearDay_Sin": np.sin(day_of_year * 2 * np.pi / 365.25),
        "YearDay_Cos": np.cos(day_of_year * 2 * np.pi / 365.25),
        "Lag_1": sales_history[-1],
        "Lag_7": sales_history[-7],
        "Lag_14": sales_history[-14],
        "Lag_28": sales_history[-28],
        "Lag_30": sales_history[-30],
        "Rolling_7": pd.Series(sales_history[-7:]).mean(),
        "Rolling_14": pd.Series(sales_history[-14:]).mean(),
        "Rolling_28": pd.Series(sales_history[-28:]).mean(),
        "Sales_Std_7": pd.Series(sales_history[-7:]).std(),
        "Sales_Std_28": pd.Series(sales_history[-28:]).std(),
        "Orders_Lag_1": orders_history[-1],
        "Orders_Lag_7": orders_history[-7],
        "Orders_Rolling_7": pd.Series(orders_history[-7:]).mean(),
        "Units_Sold_Lag_1": units_history[-1],
        "Units_Sold_Lag_7": units_history[-7],
        "Units_Sold_Rolling_7": pd.Series(units_history[-7:]).mean(),
    }

    missing_features = [name for name in features if name not in row]
    if missing_features:
        raise ValueError(
            "Prediction features do not match the saved model: "
            + ", ".join(missing_features)
        )

    model_input = pd.DataFrame([[row[name] for name in features]], columns=features)
    predicted_sales = max(0.0, float(model.predict(model_input)[0]))

    forecast_rows.append(
        {"Date": date, "Predicted_Sales": predicted_sales}
    )
    sales_history.append(predicted_sales)
    orders_history.append(estimated_orders)
    units_history.append(estimated_units)

forecast = pd.DataFrame(forecast_rows)
forecast.to_csv(DATA_DIR / "future_sales_prediction.csv", index=False)

print("Next 10 days of predicted sales:")
print(forecast.head(10).to_string(index=False))
print(f"\nPredicted sales total for the next 365 days: {forecast['Predicted_Sales'].sum():,.2f}")
print("Forecast saved to datasets/future_sales_prediction.csv")
