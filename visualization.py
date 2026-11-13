from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"

df = pd.read_csv(DATA_DIR / "cleaned_sales.csv", parse_dates=["Order_Date"])

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font="DejaVu Sans",
    rc={
        "axes.facecolor": "#C9E0EE",
        "figure.facecolor": "#ffffff",
        "grid.color": "#e8edf5",
        "grid.linewidth": 0.8,
        "axes.edgecolor": "#f1f2e5",
        "axes.labelcolor": "#252930",
        "xtick.color": "#718096",
        "ytick.color": "#718096",
    },
)

ACCENT = "#315efb"


def style_axis(ax, title, xlabel, ylabel, grid_axis="y"):
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color="#17233b", pad=14)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=9)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=9)
    ax.grid(axis=grid_axis, color="#e5eaf1")
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e5eaf2")
    ax.spines["bottom"].set_color("#e5eaf2")


def format_sales_axis(ax, axis="x"):
    formatter = FuncFormatter(lambda value, _: f"{value:,.0f}")
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)


# 1. Category-wise sales
category_sales = df.groupby("Product_Category")["Sales"].sum().sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
category_colors = sns.color_palette("husl", n_colors=len(category_sales))
bars = ax.barh(category_sales.index, category_sales.values, color=category_colors)
style_axis(ax, "Sales by Product Category", "Total sales", "Product category", grid_axis="x")
format_sales_axis(ax)
ax.bar_label(bars, labels=[f"{value:,.0f}" for value in category_sales], padding=5, fontsize=9)
ax.set_xlim(0, category_sales.max() * 1.18)
fig.tight_layout()
plt.show()

# 2. Daily sales trend with a 7-day rolling average
daily_sales = df.groupby("Order_Date")["Sales"].sum().sort_index()
rolling_sales = daily_sales.rolling(window=7, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(daily_sales.index, daily_sales.values, color=ACCENT, alpha=0.3, linewidth=1.2, label="Daily sales")
ax.plot(
    rolling_sales.index,
    rolling_sales.values,
    color=ACCENT,
    linewidth=2.5,
    label="7-day average",
)
style_axis(ax, "Daily Sales Trend", "Order date", "Sales")
format_sales_axis(ax, axis="y")
ax.legend(frameon=False, loc="upper left")
fig.autofmt_xdate()
fig.tight_layout()
plt.show()

# 3. Discount vs. sales, colored by product category
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Discount_Percent",
    y="Sales",
    hue="Product_Category",
    palette="husl",
    s=70,
    alpha=0.8,
    edgecolor="white",
    linewidth=0.6,
    ax=ax,
)
style_axis(ax, "Discount vs. Sales", "Discount (%)", "Sales")
format_sales_axis(ax, axis="y")
ax.legend(
    title="Product category",
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=False,
)
fig.tight_layout()
plt.show()

# 4. Quantity distribution
fig, ax = plt.subplots(figsize=(9, 5))
quantity_order = sorted(df["Quantity"].dropna().unique())
sns.countplot(data=df, x="Quantity", order=quantity_order, color=ACCENT, ax=ax)
style_axis(ax, "Order Quantity Distribution", "Quantity per order", "Number of orders")
fig.tight_layout()
plt.show()

# 5. Regional sales
region_sales = df.groupby("Region")["Sales"].sum().sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
region_bars = ax.barh(region_sales.index, region_sales.values, color=ACCENT)
style_axis(ax, "Sales by Region", "Total sales", "Region", grid_axis="x")
format_sales_axis(ax)
ax.bar_label(region_bars, labels=[f"{value:,.0f}" for value in region_sales], padding=5, fontsize=9)
ax.set_xlim(0, region_sales.max() * 1.18)
fig.tight_layout()
plt.show()

# 6. Sales by payment mode
payment_sales = df.groupby("Payment_Mode")["Sales"].sum().sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
payment_bars = ax.barh(
    payment_sales.index,
    payment_sales.values,
    color=sns.color_palette("husl", n_colors=len(payment_sales)),
)
style_axis(ax, "Sales by Payment Mode", "Total sales", "Payment mode", grid_axis="x")
format_sales_axis(ax)
ax.bar_label(payment_bars, labels=[f"{value:,.0f}" for value in payment_sales], padding=5, fontsize=9)
ax.set_xlim(0, payment_sales.max() * 1.18)
fig.tight_layout()
plt.show()

# 7. Customer rating distribution
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(data=df, x="Rating", bins=10, color=ACCENT, edgecolor="white", ax=ax)
style_axis(ax, "Customer Rating Distribution", "Customer rating", "Number of ratings")
ax.set_xlim(1, 5)
fig.tight_layout()
plt.show()

print("Visualizations completed.")
