from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="PredictNextGen | Sales Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17233b;
        --muted: #718096;
        --primary: #315efb;
        --teal: #0e9f9a;
        --amber: #d99018;
        --coral: #e47763;
        --paper: #f4f7fb;
        --card: #ffffff;
        --line: #e5eaf2;
    }
    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(244, 247, 251, 0.94); }
    [data-testid="stMain"] > div { padding-top: 2rem; }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] h2 { color: var(--ink); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    [data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 6px 24px rgba(27, 45, 80, 0.045);
    }
    [data-testid="stMetric"] { border-top: 3px solid var(--primary); }
    [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2) [data-testid="stMetric"] { border-top-color: var(--teal); }
    [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(3) [data-testid="stMetric"] { border-top-color: var(--amber); }
    [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(4) [data-testid="stMetric"] { border-top-color: var(--coral); }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    h1, h2, h3 { color: var(--ink); }
    h1 { letter-spacing: -0.045em; font-weight: 750; }
    h2, h3 { letter-spacing: -0.025em; }
    [data-testid="stMarkdownContainer"] h4 { color: var(--ink); font-weight: 650; }
    [data-testid="stRadio"] label p { font-weight: 550; }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stButton"] button,
    [data-testid="stDownloadButton"] button {
        border-radius: 10px;
        border-color: #d5def0;
        color: var(--primary);
        background: #ffffff;
        font-weight: 600;
    }
    [data-testid="stButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover {
        border-color: var(--primary);
        color: #ffffff;
        background: var(--primary);
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background: #e8eeff !important;
        color: #294fdb !important;
        border: 1px solid #d5dfff;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg { fill: #315efb; }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"]:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 1px var(--primary);
    }
    [data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--line); }
    .brand-mark {
        color: var(--primary);
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0 0 0.4rem 0;
    }
    .page-intro {
        color: var(--muted);
        font-size: 1rem;
        margin-top: -0.6rem;
        margin-bottom: 1.4rem;
    }
    .section-label {
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: -0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CHART_COLORS = ["#315efb", "#0e9f9a", "#d99018", "#e47763", "#7658d6", "#51a878"]
INK = "#17233b"
GRID = "#c9d3e3"
PAPER = "#a6acb1"

sns.set_theme(
    style="whitegrid",
    palette=CHART_COLORS,
    rc={
        "axes.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "axes.edgecolor": "#171b21",
        "axes.labelcolor": INK,
        "xtick.color": "#718096",
        "ytick.color": "#718096",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    },
)


@st.cache_data
def load_sales(path):
    data = pd.read_csv(path, parse_dates=["Order_Date"])
    required = {
        "Order_Date",
        "Order_ID",
        "Product_Category",
        "Product_Name",
        "Sales",
        "Quantity",
        "Region",
        "Payment_Mode",
        "Rating",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Cleaned sales data is missing columns: {', '.join(sorted(missing))}")
    return data


@st.cache_data
def load_forecast(path):
    data = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Predicted_Sales"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Forecast is missing columns: {', '.join(sorted(missing))}")
    return data.sort_values("Date")


@st.cache_data
def load_feature_data(path):
    return pd.read_csv(path, index_col="Order_Date", parse_dates=True).sort_index()


def money(value):
    return f"₹{value:,.0f}"


LIVE_ACTUAL_COLUMNS = ["Date", "Actual_Sales", "Orders", "Units_Sold"]
LIVE_PREDICTION_COLUMNS = [
    "Date",
    "Predicted_Sales",
    "Latest_Sales_Input",
    "Expected_Orders",
    "Expected_Units",
    "Generated_At",
]


def normalize_live_upload(uploaded_data):
    source = uploaded_data.copy()
    columns = {str(column).strip().lower(): column for column in source.columns}
    date_column = columns.get("date") or columns.get("order_date")
    sales_column = columns.get("actual_sales") or columns.get("sales")
    if date_column is None or sales_column is None:
        raise ValueError("CSV needs Date (or Order_Date) and Sales (or Actual_Sales) columns.")

    source[date_column] = pd.to_datetime(source[date_column], errors="coerce")
    source[sales_column] = pd.to_numeric(source[sales_column], errors="coerce")
    if source[[date_column, sales_column]].isna().any().any():
        raise ValueError("CSV contains missing or invalid dates or sales values.")
    order_id_column = columns.get("order_id")
    quantity_column = columns.get("quantity")

    if order_id_column is not None and quantity_column is not None:
        source[quantity_column] = pd.to_numeric(source[quantity_column], errors="coerce")
        if source[[date_column, sales_column, quantity_column, order_id_column]].isna().any().any():
            raise ValueError("Transaction CSV contains missing or invalid dates, sales, quantities, or order IDs.")
        if (source[sales_column] < 0).any() or (source[quantity_column] < 0).any():
            raise ValueError("Sales and quantities must be zero or greater.")
        result = (
            source.groupby(date_column)
            .agg(
                Actual_Sales=(sales_column, "sum"),
                Orders=(order_id_column, "nunique"),
                Units_Sold=(quantity_column, "sum"),
            )
            .reset_index()
            .rename(columns={date_column: "Date"})
        )
    else:
        orders_column = columns.get("orders")
        units_column = columns.get("units_sold")
        if orders_column is None or units_column is None:
            raise ValueError(
                "Daily CSV also needs Orders and Units_Sold, or upload transaction rows "
                "with Order_ID and Quantity."
            )
        source[orders_column] = pd.to_numeric(source[orders_column], errors="coerce")
        source[units_column] = pd.to_numeric(source[units_column], errors="coerce")
        required = [date_column, sales_column, orders_column, units_column]
        if source[required].isna().any().any():
            raise ValueError("Daily CSV contains missing or invalid date, sales, order, or unit values.")
        if (source[[sales_column, orders_column, units_column]] < 0).any().any():
            raise ValueError("Sales, orders, and units must be zero or greater.")
        result = source[required].rename(
            columns={
                date_column: "Date",
                sales_column: "Actual_Sales",
                orders_column: "Orders",
                units_column: "Units_Sold",
            }
        )

    result["Date"] = pd.to_datetime(result["Date"]).dt.normalize()
    result = result[LIVE_ACTUAL_COLUMNS]
    result = result.sort_values("Date").drop_duplicates("Date", keep="last")
    if result.empty:
        raise ValueError("CSV does not contain any valid sales records.")
    return result


def merge_by_date(existing, updates, columns):
    combined = pd.concat([existing[columns], updates[columns]], ignore_index=True)
    return (
        combined.drop_duplicates("Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )


def load_live_table(path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    data = pd.read_csv(path)
    missing = set(columns).difference(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    data = data[columns].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    numeric_columns = [column for column in columns if column not in ("Date", "Generated_At")]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[columns].isna().any().any():
        raise ValueError(f"{path.name} contains missing or invalid values.")
    return data.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def make_daily_history(transactions, live_actuals):
    daily = (
        transactions.groupby("Order_Date")
        .agg(
            Sales=("Sales", "sum"),
            Orders=("Order_ID", "nunique"),
            Units_Sold=("Quantity", "sum"),
        )
        .rename_axis("Date")
    )
    if not live_actuals.empty:
        overrides = live_actuals.set_index("Date").rename(
            columns={"Actual_Sales": "Sales"}
        )[["Sales", "Orders", "Units_Sold"]]
        daily = pd.concat([daily, overrides]).groupby(level=0).last()
    return daily.sort_index().asfreq("D", fill_value=0)


def predict_for_date(
    history,
    model,
    features,
    target_date,
    latest_sales_input,
    expected_orders,
    expected_units,
):
    target_date = pd.Timestamp(target_date).normalize()
    sales_history = history["Sales"].astype(float).tolist()
    sales_history[-1] = float(latest_sales_input)
    orders_history = history["Orders"].astype(float).tolist()
    units_history = history["Units_Sold"].astype(float).tolist()
    dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), target_date, freq="D")

    for date in dates:
        day_of_week = date.dayofweek
        day_of_year = date.dayofyear
        orders_for_day = expected_orders if date == target_date else orders_history[-7]
        units_for_day = expected_units if date == target_date else units_history[-7]
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
                "Saved model features do not match the predictor: "
                + ", ".join(missing_features)
            )
        model_input = pd.DataFrame([[row[name] for name in features]], columns=features)
        predicted_sales = max(0.0, float(model.predict(model_input)[0]))
        sales_history.append(predicted_sales)
        orders_history.append(float(orders_for_day))
        units_history.append(float(units_for_day))

    return predicted_sales


def bar_chart(data, title, xlabel, ylabel, horizontal=False):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [CHART_COLORS[index % len(CHART_COLORS)] for index in range(len(data))]
    max_value = float(data.max()) if not data.empty else 0
    ax.set_axisbelow(True)
    if horizontal:
        bars = ax.barh(data.index.astype(str), data.values, color=colors)
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f"{value:,.0f}" for value in data.values], padding=5, fontsize=8, color="#52627a")
        if max_value > 0:
            ax.set_xlim(0, max_value * 1.18)
    else:
        bars = ax.bar(data.index.astype(str), data.values, color=colors)
        ax.tick_params(axis="x", rotation=25)
        ax.bar_label(bars, labels=[f"{value:,.0f}" for value in data.values], padding=3, fontsize=8, color="#52627a")
        if max_value > 0:
            ax.set_ylim(0, max_value * 1.18)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x" if horizontal else "y", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=6)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


sales_path = DATA_DIR / "cleaned_sales.csv"
forecast_path = DATA_DIR / "future_sales_prediction.csv"
feature_data_path = DATA_DIR / "feature_engineered.csv"
raw_sales_path = DATA_DIR / "ecommerce_sales_prediction_practice.csv"
model_path = MODEL_DIR / "sales_model.pkl"
features_path = MODEL_DIR / "features.pkl"
metrics_path = MODEL_DIR / "model_metrics.pkl"
live_actuals_path = DATA_DIR / "live_sales.csv"
live_predictions_path = DATA_DIR / "live_predictions.csv"
missing_files = [path.name for path in (sales_path,) if not path.exists()]
if missing_files:
    st.error(
        "Required data file(s) not found: "
        + ", ".join(missing_files)
        + ". Run data_cleaning.py first from the project folder."
    )
    st.stop()

try:
    sales = load_sales(str(sales_path))
    forecast = (
        load_forecast(str(forecast_path))
        if forecast_path.exists()
        else pd.DataFrame(columns=["Date", "Predicted_Sales"])
    )
except (ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load project data: {error}")
    st.stop()

if sales.empty:
    st.error("The cleaned sales dataset does not contain any rows.")
    st.stop()

sales["Sales"] = pd.to_numeric(sales["Sales"], errors="coerce")
sales["Quantity"] = pd.to_numeric(sales["Quantity"], errors="coerce")
sales["Rating"] = pd.to_numeric(sales["Rating"], errors="coerce")
forecast["Predicted_Sales"] = pd.to_numeric(forecast["Predicted_Sales"], errors="coerce")
sales = sales.dropna(subset=["Order_Date", "Sales"])
if not forecast.empty:
    forecast = forecast.dropna(subset=["Date", "Predicted_Sales"])
if sales.empty:
    st.error("No valid sales records were found in the cleaned dataset.")
    st.stop()

all_sales = sales.copy()

st.markdown('<p class="brand-mark">PredictNextGen &nbsp;·&nbsp; BUSINESS INTELLIGENCE</p>', unsafe_allow_html=True)
st.title("Sales performance, clearly.")
st.markdown(
    '<p class="page-intro">Explore the story behind your sales, understand the forecast, '
    "and keep every step of the data pipeline in view.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<p class="brand-mark">PREDICTNEXTGEN</p>', unsafe_allow_html=True)
    st.markdown("### Sales intelligence")
    st.caption("E-commerce forecasting workspace")
    st.divider()
    st.markdown("#### Workspace")
    page = st.radio(
        "Navigate",
        [
            "Executive Overview",
            "Sales EDA",
            "Forecast",
            "Model & Method",
            "Pipeline & Data Quality",
            "Live Prediction & Accuracy",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("## Filters")
    st.caption("These filters apply to historical sales analysis.")
    min_date = all_sales["Order_Date"].min().date()
    max_date = all_sales["Order_Date"].max().date()
    selected_dates = st.date_input(
        "Order date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates

    category_options = sorted(all_sales["Product_Category"].dropna().unique().tolist())
    selected_categories = st.multiselect(
        "Product categories",
        options=category_options,
        default=category_options,
        help="Clear the selection to show no categories.",
    )
    region_options = sorted(all_sales["Region"].dropna().unique().tolist())
    selected_regions = st.multiselect(
        "Regions",
        options=region_options,
        default=region_options,
        help="Clear the selection to show no regions.",
    )

    sales = all_sales.loc[
        all_sales["Order_Date"].dt.date.between(start_date, end_date)
        & all_sales["Product_Category"].isin(selected_categories)
        & all_sales["Region"].isin(selected_regions)
    ].copy()

    st.divider()
    st.caption("DATA WORKFLOW")
    st.code(
        "raw data\n  → data_cleaning.py\n  → eda.py\n  → feature_engineering.py\n  → train_model.py\n  → predict.py\n  → app.py",
        language="text",
    )

if sales.empty and page in ("Executive Overview", "Sales EDA"):
    st.warning("No transactions match the selected filters. Widen the date range or select more categories/regions.")
    st.stop()

sales_daily = sales.groupby("Order_Date")["Sales"].sum().sort_index()
monthly_sales = sales.set_index("Order_Date")["Sales"].resample("MS").sum()

if page == "Executive Overview":
    st.markdown('<p class="section-label">Executive overview</p>', unsafe_allow_html=True)
    st.header("Business snapshot")
    total_sales = sales["Sales"].sum()
    order_count = sales["Order_ID"].nunique()
    average_order_value = total_sales / order_count if order_count else 0
    average_rating = sales["Rating"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Historical sales", money(total_sales))
    k2.metric("Orders", f"{order_count:,}")
    k3.metric("Average order value", money(average_order_value))
    k4.metric("Average rating", f"{average_rating:.1f} / 5" if pd.notna(average_rating) else "N/A")

    st.caption(
        f"Observed period: {sales['Order_Date'].min():%d %b %Y} – "
        f"{sales['Order_Date'].max():%d %b %Y} · "
        f"{len(sales):,} filtered transaction records"
    )

    left, right = st.columns([1.6, 1])
    with left:
        st.markdown("#### Daily sales")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_axisbelow(True)
        ax.plot(sales_daily.index, sales_daily.values, color="#a9bafa", alpha=0.58, linewidth=1)
        ax.plot(
            sales_daily.rolling(7, min_periods=1).mean().index,
            sales_daily.rolling(7, min_periods=1).mean().values,
            color="#315efb",
            linewidth=2.5,
            label="7-day average",
        )
        ax.set_ylabel("Sales (₹)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="both", length=0, pad=6)
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        category_sales = sales.groupby("Product_Category")["Sales"].sum().sort_values()
        bar_chart(category_sales, "Sales by category", "Sales (₹)", "Category", horizontal=True)

elif page == "Sales EDA":
    st.markdown('<p class="section-label">Understand the business</p>', unsafe_allow_html=True)
    st.header("Sales exploration")
    st.markdown(
        '<p class="page-intro">Compare product, regional, and payment performance over your selected period.</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        category_sales = sales.groupby("Product_Category")["Sales"].sum().sort_values(ascending=False)
        bar_chart(category_sales, "Sales by product category", "Category", "Sales (₹)")
    with col2:
        region_sales = sales.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        bar_chart(region_sales, "Sales by region", "Region", "Sales (₹)")

    col3, col4 = st.columns(2)
    with col3:
        payment_sales = sales.groupby("Payment_Mode")["Sales"].sum().sort_values(ascending=False)
        bar_chart(payment_sales, "Sales by payment mode", "Payment mode", "Sales (₹)", horizontal=True)
    with col4:
        product_sales = sales.groupby("Product_Name")["Sales"].sum().nlargest(10).sort_values()
        bar_chart(product_sales, "Top 10 products by sales", "Sales (₹)", "Product", horizontal=True)

    st.markdown("#### Monthly sales")
    monthly_by_period = monthly_sales.copy()
    monthly_by_period.index = monthly_by_period.index.strftime("%b %Y")
    bar_chart(monthly_by_period, "Monthly sales trend", "Month", "Sales (₹)")

    with st.expander("Dataset summary"):
        st.write(
            {
                "Rows": int(len(sales)),
                "Columns": int(sales.shape[1]),
                "Missing values": int(sales.isna().sum().sum()),
                "Unique products": int(sales["Product_Name"].nunique()),
                "Regions": int(sales["Region"].nunique()),
            }
        )
        st.dataframe(sales.describe(include="all").T, use_container_width=True)

elif page == "Forecast":
    st.markdown('<p class="section-label">Forward view</p>', unsafe_allow_html=True)
    st.header("Sales forecast")
    if forecast.empty:
        st.warning(
            "The forecast file is not available or contains no valid rows. "
            "Run predict.py after training the model to generate it."
        )
        st.stop()

    forecast_start = forecast["Date"].min()
    forecast_end = forecast["Date"].max()
    today = pd.Timestamp.today().normalize()

    if forecast_start <= today:
        st.warning(
            f"This forecast starts on {forecast_start:%d %b %Y}, which is not a future date."
        )
    if forecast_start <= all_sales["Order_Date"].max():
        st.warning("Forecast dates overlap with historical data; regenerate the forecast pipeline.")

    total_forecast = forecast["Predicted_Sales"].sum()
    daily_average = forecast["Predicted_Sales"].mean()
    best_day = forecast.loc[forecast["Predicted_Sales"].idxmax()]
    f1, f2, f3 = st.columns(3)
    f1.metric("Forecast total", money(total_forecast))
    f2.metric("Average forecast / day", money(daily_average))
    f3.metric("Highest forecast day", f"{best_day['Date']:%d %b %Y} · {money(best_day['Predicted_Sales'])}")
    st.caption(f"Forecast horizon: {forecast_start:%d %b %Y} – {forecast_end:%d %b %Y}")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_axisbelow(True)
    ax.plot(forecast["Date"], forecast["Predicted_Sales"], color="#a9bafa", linewidth=1.5)
    ax.plot(
        forecast["Date"],
        forecast["Predicted_Sales"].rolling(7, min_periods=1).mean(),
        color="#315efb",
        linewidth=2.5,
        label="7-day forecast average",
    )
    ax.set_ylabel("Predicted sales (₹)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("#### Forecast records")
    st.dataframe(forecast, use_container_width=True, hide_index=True)
    st.download_button(
        "Download forecast CSV",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name="future_sales_prediction.csv",
        mime="text/csv",
    )
    st.info(
        "This is a recursive forecast: each predicted day is used as input for later days. "
        "Long-range forecasts are uncertain and should be treated as estimates, not guarantees."
    )
    st.caption("Historical date, category, and region filters apply to sales analysis only; this forecast is for all sales combined.")

elif page == "Model & Method":
    st.markdown('<p class="section-label">Machine learning</p>', unsafe_allow_html=True)
    st.header("Model performance & methodology")
    if metrics_path.exists():
        metrics = joblib.load(metrics_path)
        m1, m2, m3 = st.columns(3)
        m1.metric("Holdout MAE", money(metrics["mae"]))
        m2.metric("Holdout RMSE", money(metrics["rmse"]))
        m3.metric("Holdout R²", f"{metrics['r2']:.3f}")
        st.caption(
            f"Model: {metrics['model']} · "
            f"Training: through {pd.Timestamp(metrics['train_end']):%d %b %Y} · "
            f"Test: {pd.Timestamp(metrics['test_start']):%d %b %Y} to "
            f"{pd.Timestamp(metrics['test_end']):%d %b %Y}"
        )
        st.write(
            f"Simple 7-day baseline MAE: **{money(metrics['baseline_mae'])}**. "
            f"{'The model beat this baseline.' if metrics['mae'] < metrics['baseline_mae'] else 'The model did not beat this baseline.'}"
        )
        if "baseline_r2" in metrics:
            st.caption(
                f"7-day baseline R²: {metrics['baseline_r2']:.3f}. "
                "A negative R² means predictions are worse than using the test-period average."
            )
    else:
        st.info(
            "Holdout metrics are not saved yet. Run train_model.py once to create them; "
            "the dashboard will then show measured test results."
        )

    st.markdown("#### What the evaluation means")
    st.markdown(
        "- **Time-based split:** older dates train the model; the latest 20% are held out for testing.\n"
        "- **MAE:** average absolute prediction error, in sales units; lower is better.\n"
        "- **RMSE:** error that penalizes large misses more strongly; lower is better.\n"
        "- **R²:** compares prediction error with predicting the test-period average; a negative value means the model is worse than that reference.\n"
        "- **7-day baseline:** predicts this week's sales using the same weekday from last week."
    )

    if model_path.exists() and features_path.exists():
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=features).nlargest(10).sort_values()
            bar_chart(importance, "Top 10 model feature importances", "Importance", "Feature", horizontal=True)
            st.caption("Feature importance shows model reliance, not proof of cause and effect.")

    if feature_data_path.exists():
        feature_data = load_feature_data(str(feature_data_path))
        st.markdown("#### Feature engineering preview")
        st.caption(f"{len(feature_data):,} daily rows · {feature_data.shape[1] - 1} input features")
        st.dataframe(feature_data.head(10), use_container_width=True)
    else:
        st.info("Feature preview is unavailable. Run feature_engineering.py to generate it.")

elif page == "Pipeline & Data Quality":
    st.markdown('<p class="section-label">Trust & transparency</p>', unsafe_allow_html=True)
    st.header("Pipeline & data quality")
    st.markdown(
        '<p class="page-intro">Track generated artifacts, understand each Python module, '
        "and compare source data with the cleaned dataset.</p>",
        unsafe_allow_html=True,
    )

    pipeline_steps = [
        ("1 · Clean", "data_cleaning.py", raw_sales_path, sales_path),
        ("2 · Explore", "eda.py", sales_path, None),
        ("3 · Engineer features", "feature_engineering.py", sales_path, feature_data_path),
        ("4 · Train and evaluate", "train_model.py", feature_data_path, model_path),
        ("5 · Forecast", "predict.py", model_path, forecast_path),
        ("6 · Dashboard", "app.py", sales_path, None),
    ]
    pipeline_status = pd.DataFrame(
        [
            {
                "Stage": stage,
                "Script": script,
                "Input available": input_path.exists(),
                "Output": output_path.name if output_path else "Console analysis",
                "Output available": output_path.exists() if output_path else True,
            }
            for stage, script, input_path, output_path in pipeline_steps
        ]
    )
    st.markdown("#### Workflow status")
    st.dataframe(pipeline_status, use_container_width=True, hide_index=True)

    module_inventory = pd.DataFrame(
        [
            {
                "Python module": "data_cleaning.py",
                "Role": "Validate, clean, and save transaction data",
            },
            {
                "Python module": "eda.py",
                "Role": "Summarize sales by category, month, product, and region",
            },
            {
                "Python module": "feature_engineering.py",
                "Role": "Build daily calendar, lag, and rolling sales features",
            },
            {
                "Python module": "train_model.py",
                "Role": "Time-split evaluation, Random Forest training, and model export",
            },
            {
                "Python module": "predict.py",
                "Role": "Generate a recursive 365-day sales forecast",
            },
            {
                "Python module": "visualization.py",
                "Role": "Render standalone exploratory charts",
            },
            {
                "Python module": "app.py",
                "Role": "Interactive sales dashboard, live predictions, and accuracy tracking",
            },
            {
                "Python module": "tempCodeRunnerFile.py",
                "Role": "Commented-out scratch code; not part of the production pipeline",
            },
        ]
    )
    st.markdown("#### Python module guide")
    st.dataframe(module_inventory, use_container_width=True, hide_index=True)

    artifact_status = pd.DataFrame(
        [
            {"Artifact": path.name, "Available": path.exists()}
            for path in (
                raw_sales_path,
                sales_path,
                feature_data_path,
                model_path,
                features_path,
                metrics_path,
                forecast_path,
                live_actuals_path,
                live_predictions_path,
            )
        ]
    )
    st.markdown("#### Project artifacts")
    st.dataframe(artifact_status, use_container_width=True, hide_index=True)

    if raw_sales_path.exists():
        try:
            raw_sales = pd.read_csv(raw_sales_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read the raw sales dataset: {error}")
        else:
            if "Order_ID" not in raw_sales.columns:
                st.warning("The raw dataset does not contain Order_ID, so duplicate-order checks are unavailable.")
            else:
                raw_duplicates = int(raw_sales.duplicated(subset="Order_ID").sum())
                cleaned_duplicates = int(all_sales.duplicated(subset="Order_ID").sum())
                raw_missing = int(raw_sales.isna().sum().sum())
                cleaned_missing = int(all_sales.isna().sum().sum())

                q1, q2, q3, q4, q5 = st.columns(5)
                q1.metric("Raw records", f"{len(raw_sales):,}")
                q2.metric("Cleaned records", f"{len(all_sales):,}")
                q3.metric("Raw duplicate order IDs", f"{raw_duplicates:,}")
                q4.metric("Cleaned duplicate order IDs", f"{cleaned_duplicates:,}")
                q5.metric("Missing cells · raw → cleaned", f"{raw_missing:,} → {cleaned_missing:,}")
                st.caption(
                    "The cleaning script drops duplicate Order_ID values, invalid dates and "
                    "invalid required numeric values; it fills optional category, region, "
                    "payment, and discount values."
                )

                missing_profile = pd.DataFrame(
                    {
                        "Raw missing": raw_sales.isna().sum(),
                        "Cleaned missing": all_sales.reindex(columns=raw_sales.columns).isna().sum(),
                    }
                )
                missing_profile = missing_profile.loc[missing_profile.sum(axis=1) > 0]
                st.markdown("#### Missing values by column")
                if missing_profile.empty:
                    st.success("No missing values remain in the source or cleaned data.")
                else:
                    st.dataframe(
                        missing_profile.sort_values("Raw missing", ascending=False),
                        use_container_width=True,
                    )

                st.markdown("#### Cleaned data preview")
                st.dataframe(all_sales.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("The raw practice dataset is unavailable, so source-to-cleaning comparisons cannot be shown.")

else:
    st.markdown('<p class="section-label">Interactive forecasting</p>', unsafe_allow_html=True)
    st.header("Live sales prediction & accuracy")
    st.markdown(
        '<p class="page-intro">Enter assumptions for a future date, record actual sales as they arrive, '
        "and track real-world prediction error.</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "This page uses the saved model and the latest available daily history. "
        "CSV or manual actuals update the history and live score after you submit them."
    )

    try:
        live_actuals = load_live_table(live_actuals_path, LIVE_ACTUAL_COLUMNS)
        live_predictions = load_live_table(live_predictions_path, LIVE_PREDICTION_COLUMNS)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not load live dashboard records: {error}")
        st.stop()

    st.markdown("#### Add real sales data")
    csv_tab, manual_tab = st.tabs(["Import CSV", "Enter one day"])
    with csv_tab:
        st.caption(
            "Upload daily rows with Date, Sales, Orders, Units_Sold, or transaction rows "
            "with Order_Date, Order_ID, Quantity, Sales."
        )
        st.download_button(
            "Download daily CSV template",
            data=pd.DataFrame(columns=LIVE_ACTUAL_COLUMNS).to_csv(index=False).encode("utf-8"),
            file_name="live_sales_template.csv",
            mime="text/csv",
        )
        with st.form("live_sales_csv_form"):
            live_upload = st.file_uploader("Sales CSV", type=["csv"], key="live_sales_upload")
            import_submitted = st.form_submit_button("Import and update history")
        if import_submitted:
            if live_upload is None:
                st.error("Choose a CSV file before importing.")
            else:
                try:
                    uploaded_rows = normalize_live_upload(pd.read_csv(live_upload))
                    if (uploaded_rows["Date"] > pd.Timestamp.today().normalize()).any():
                        raise ValueError("Actual sales data cannot contain future dates.")
                    saved_rows = merge_by_date(live_actuals, uploaded_rows, LIVE_ACTUAL_COLUMNS)
                    saved_rows.to_csv(live_actuals_path, index=False)
                except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
                    st.error(f"Could not import sales data: {error}")
                else:
                    st.success(f"Updated {len(uploaded_rows):,} daily record(s).")
                    st.rerun()

    with manual_tab:
        st.caption("Record actual daily sales. Entries are saved locally and used in future predictions.")
        with st.form("live_sales_manual_form"):
            actual_date = st.date_input(
                "Actual sales date",
                value=pd.Timestamp.today().date(),
                max_value=pd.Timestamp.today().date(),
                key="live_actual_date",
            )
            actual_sales_value = st.number_input(
                "Actual sales (₹)", min_value=0.0, value=0.0, step=100.0
            )
            actual_orders = st.number_input("Actual orders", min_value=0, value=0, step=1)
            actual_units = st.number_input("Actual units sold", min_value=0, value=0, step=1)
            manual_submitted = st.form_submit_button("Save actual sales")
        if manual_submitted:
            entered_actual = pd.DataFrame(
                [
                    {
                        "Date": pd.Timestamp(actual_date),
                        "Actual_Sales": actual_sales_value,
                        "Orders": actual_orders,
                        "Units_Sold": actual_units,
                    }
                ]
            )
            saved_rows = merge_by_date(live_actuals, entered_actual, LIVE_ACTUAL_COLUMNS)
            saved_rows.to_csv(live_actuals_path, index=False)
            st.success("Actual sales saved and daily history updated.")
            st.rerun()

    try:
        live_actuals = load_live_table(live_actuals_path, LIVE_ACTUAL_COLUMNS)
        live_predictions = load_live_table(live_predictions_path, LIVE_PREDICTION_COLUMNS)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not refresh live dashboard records: {error}")
        st.stop()

    st.divider()
    st.markdown("#### Model accuracy")
    if metrics_path.exists():
        metrics = joblib.load(metrics_path)
        metric_cols = st.columns(5)
        metric_cols[0].metric("Holdout MAE", money(metrics["mae"]))
        metric_cols[1].metric("Holdout RMSE", money(metrics["rmse"]))
        metric_cols[2].metric("Holdout R²", f"{metrics['r2']:.3f}")
        metric_cols[3].metric("7-day baseline MAE", money(metrics["baseline_mae"]))
        if "baseline_r2" in metrics:
            metric_cols[4].metric("7-day baseline R²", f"{metrics['baseline_r2']:.3f}")
        st.caption(
            f"Time-based holdout · test dates {pd.Timestamp(metrics['test_start']):%d %b %Y} "
            f"to {pd.Timestamp(metrics['test_end']):%d %b %Y}. "
            f"{'The model beat' if metrics['mae'] < metrics['baseline_mae'] else 'The model did not beat'} "
            "the 7-day baseline on this test period."
        )
    else:
        st.info("Saved holdout metrics are not available yet. Run train_model.py to evaluate the model.")

    scored = live_predictions.merge(
        live_actuals[["Date", "Actual_Sales"]],
        on="Date",
        how="inner",
    )
    if scored.empty:
        st.info(
            "Live accuracy will appear after you save actual sales for a date that has a saved prediction. "
            "Holdout metrics above remain available as the initial benchmark."
        )
    else:
        errors = scored["Predicted_Sales"] - scored["Actual_Sales"]
        live_mae = float(errors.abs().mean())
        live_rmse = float(np.sqrt(np.square(errors).mean()))
        live_r2 = None
        if len(scored) > 1 and float(np.square(scored["Actual_Sales"] - scored["Actual_Sales"].mean()).sum()) > 0:
            live_r2 = 1 - float(
                np.square(errors).sum()
                / np.square(scored["Actual_Sales"] - scored["Actual_Sales"].mean()).sum()
            )
        live_metrics = st.columns(3)
        live_metrics[0].metric("Live MAE", money(live_mae))
        live_metrics[1].metric("Live RMSE", money(live_rmse))
        live_metrics[2].metric("Live R²", f"{live_r2:.3f}" if live_r2 is not None else "N/A")
        scored["Absolute_Error"] = errors.abs()
        st.dataframe(
            scored[["Date", "Predicted_Sales", "Actual_Sales", "Absolute_Error"]]
            .sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown("#### Predict a future date")
    if not model_path.exists() or not features_path.exists():
        st.warning("The trained model is unavailable. Run train_model.py before making predictions.")
    else:
        history = make_daily_history(all_sales, live_actuals)
        if len(history) < 30:
            st.warning("At least 30 days of sales history are required for prediction.")
        else:
            min_forecast_date = (history.index[-1] + pd.Timedelta(days=1)).date()
            max_forecast_date = (history.index[-1] + pd.Timedelta(days=365)).date()
            with st.form("live_prediction_form"):
                prediction_date = st.date_input(
                    "Forecast date",
                    value=min_forecast_date,
                    min_value=min_forecast_date,
                    max_value=max_forecast_date,
                )
                latest_sales_input = st.number_input(
                    "Latest daily sales (₹)",
                    min_value=0.0,
                    value=float(history["Sales"].iloc[-1]),
                    step=100.0,
                    help="Scenario value for the most recent day. It directly updates the model's lag and rolling features for this prediction.",
                )
                expected_orders = st.number_input(
                    "Expected orders",
                    min_value=0,
                    value=max(0, int(history["Orders"].iloc[-7])),
                    step=1,
                )
                expected_units = st.number_input(
                    "Expected units sold",
                    min_value=0,
                    value=max(0, int(history["Units_Sold"].iloc[-7])),
                    step=1,
                )
                st.caption(
                    "The saved model learns from date and past sales/order/unit history. "
                    "Expected orders and units are recorded as scenario context; only latest daily sales "
                    "is used as a direct what-if input for the selected forecast."
                )
                prediction_submitted = st.form_submit_button("Predict sales")
            if prediction_submitted:
                try:
                    model = joblib.load(model_path)
                    model_features = joblib.load(features_path)
                    predicted_sales = predict_for_date(
                        history,
                        model,
                        model_features,
                        prediction_date,
                        latest_sales_input,
                        expected_orders,
                        expected_units,
                    )
                except ValueError as error:
                    st.error(f"Could not generate prediction: {error}")
                else:
                    new_prediction = pd.DataFrame(
                        [
                            {
                                "Date": pd.Timestamp(prediction_date),
                                "Predicted_Sales": predicted_sales,
                                "Latest_Sales_Input": latest_sales_input,
                                "Expected_Orders": expected_orders,
                                "Expected_Units": expected_units,
                                "Generated_At": pd.Timestamp.now().isoformat(),
                            }
                        ]
                    )
                    saved_predictions = merge_by_date(
                        live_predictions,
                        new_prediction,
                        LIVE_PREDICTION_COLUMNS,
                    )
                    saved_predictions.to_csv(live_predictions_path, index=False)
                    st.success(
                        f"Predicted sales for {prediction_date:%d %b %Y}: {money(predicted_sales)}"
                    )
                    st.caption(
                        "The prediction is saved. When actual sales for this date are imported "
                        "or entered, live MAE, RMSE, and R² will update."
                    )

    if not live_actuals.empty:
        with st.expander("Saved live sales records"):
            st.dataframe(
                live_actuals.sort_values("Date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    st.info(
        "The trained model stays fixed between training runs. New actual data immediately updates "
        "the forecasting history and live accuracy; run train_model.py to refresh model weights."
    )

st.divider()
st.caption(
    "PredictNext · E-commerce Sales Forecasting Project · "
    "Forecasts are estimates and should be validated against real future results."
)
