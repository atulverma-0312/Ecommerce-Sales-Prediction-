from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"

df = pd.read_csv(DATA_DIR / "cleaned_sales.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nStatistical Summary:")
print(df.describe())

print("\nCategory Wise Sales:")
print(
    df.groupby("Product_Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nMonthly Sales:")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
print(
    df.groupby(df["Order_Date"].dt.to_period("M"))["Sales"].sum()
)

print("\nTop 10 Products:")
print(
    df.groupby("Product_Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nRegion Wise Sales:")
print(
    df.groupby("Region")["Sales"].sum()
)

print("\nEDA Completed!")