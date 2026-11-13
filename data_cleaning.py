from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"

# First Load the dataset
df = pd.read_csv(DATA_DIR / "ecommerce_sales_prediction_practice.csv")

# This is the Dataset information
print(df.head())
print(df.info())

# Missing values
print("Missing Values:")
print(df.isnull().sum())

# Find Duplicate records
print("Duplicates:", df.duplicated().sum())

# Remove duplicates records
df = df.drop_duplicates(subset="Order_ID")

# Correct data types
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"], errors="coerce"
)

numeric_cols = [
    "Quantity", "Unit_Price",
    "Discount_Percent", "Sales"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid records
df = df.dropna(
    subset=["Order_Date", "Quantity", "Unit_Price", "Sales"]
)

df = df[
    (df["Quantity"] > 0) &
    (df["Unit_Price"] >= 0) &
    (df["Sales"] >= 0)
]

# Fill optional missing values
df["Product_Category"] = df["Product_Category"].fillna("Unknown")
df["Region"] = df["Region"].fillna("Unknown")
df["Payment_Mode"] = df["Payment_Mode"].fillna("Unknown")
df["Discount_Percent"] = df["Discount_Percent"].fillna(0)

# Save cleaned dataset
df.to_csv(DATA_DIR / "cleaned_sales.csv", index=False)

print("Data Cleaning Completed!")
print(df.head())