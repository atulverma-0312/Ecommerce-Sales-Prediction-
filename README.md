
# 📈 E-Commerce Sales Prediction & Forecasting

### An End-to-End Data Analytics and Machine Learning Project

<p align="center">
  <b>Transforming E-Commerce Transaction Data into Business Insights and Future Sales Predictions</b>
</p>

---

## 📌 Project Overview

**E-Commerce Sales Prediction** is an end-to-end Data Analytics and Machine Learning project designed to analyze historical e-commerce transactions, discover meaningful business insights, and forecast future sales trends using machine learning.

The project follows a structured data science workflow, starting with data cleaning and exploratory data analysis (EDA), followed by feature engineering, model development, performance evaluation, and future sales forecasting.

An interactive **Streamlit Business Intelligence Dashboard** is also developed to help users explore sales performance, understand forecasting results, monitor model performance, and analyze data quality.

The objective is to transform raw transaction data into actionable insights that can support better business planning and data-driven decision-making.

## 🎯 Project Objectives

- Analyze historical e-commerce sales performance.
- Identify top-performing products and product categories.
- Understand monthly, daily, and regional sales trends.
- Perform data cleaning and preprocessing.
- Create meaningful time-series forecasting features.
- Train and compare multiple machine learning models.
- Evaluate model performance using standard regression metrics.
- Forecast future sales for the next 365 days.
- Develop an interactive dashboard for business intelligence.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Dashboard | Streamlit |
| Development Environment | VS Code, Jupyter Notebook |
| Version Control | Git & GitHub |

## 📂 Project Structure

```text
Ecommerce-Sales-Prediction/
│
├── .vscode/
│
├── datasets/
│   ├── ecommerce_sales_prediction_practice.csv
│   ├── cleaned_sales.csv
│   ├── feature_engineered.csv
│   ├── future_sales_prediction.csv
│   ├── live_sales.csv
│   └── live_predictions.csv
│
├── models/
│   ├── sales_model.pkl
│   ├── features.pkl
│   └── model_metrics.pkl
│
├── notebook/
│   └── Jupyter Notebook files
│
├── data_cleaning.py
├── eda.py
├── feature_engineering.py
├── train_model.py
├── predict.py
├── visualization.py
├── app.py
│
├── requirements.txt
└── README.md
```

*Note: The datasets and model artifacts listed above include generated pipeline outputs. They are created or populated as the workflow runs.*

## 🔄 Project Workflow

The project follows a complete data science pipeline:

```text
Raw E-Commerce Dataset
          |
          ▼
    Data Cleaning
          |
          ▼
 Exploratory Data Analysis
          |
          ▼
   Feature Engineering
          |
          ▼
   Model Training
          |
          ▼
 Model Evaluation & Selection
          |
          ▼
   Future Sales Forecast
          |
          ▼
 Interactive Streamlit Dashboard
```

### 1. Data Cleaning & Preprocessing

File: `data_cleaning.py`

The raw dataset is processed to improve data quality and consistency.

Key operations:
- Inspect dataset structure and data types.
- Identify missing values.
- Detect and remove duplicate Order IDs.
- Convert order dates into datetime format.
- Handle invalid numeric values.
- Remove invalid sales and quantity records.
- Fill missing categorical values.
- Generate a cleaned dataset.

**Output:** `datasets/cleaned_sales.csv`

### 2. Exploratory Data Analysis (EDA)

File: `eda.py`

EDA is performed to understand sales patterns and identify important business trends.

Analysis includes:
- Dataset structure and statistical summary.
- Product category-wise sales.
- Monthly sales performance.
- Top 10 products by sales.
- Region-wise sales analysis.

### 3. Feature Engineering

File: `feature_engineering.py`

Historical transaction data is converted into daily time-series forecasting features.

**Calendar Features**
- Year, Quarter, Month and Day.
- Day of the week.
- Weekend indicators.
- Month-start and month-end indicators.
- Cyclical weekday and yearly date features.

**Historical Sales Features**
- Lag features: 1, 7, 14, 28 and 30 days.
- Rolling averages: 7, 14 and 28 days.
- Rolling standard deviations.
- Historical order and quantity features.

These features help machine learning models understand historical sales behavior and temporal patterns.

**Output:** `datasets/feature_engineered.csv`

### 4. Machine Learning Model Development

File: `train_model.py`

Multiple regression algorithms are evaluated to identify the most suitable model for sales forecasting.

**Models Implemented**

1. Random Forest Regressor
2. Log-Transformed Random Forest Regressor
3. Histogram-Based Gradient Boosting Regressor

**Model Selection Strategy**

- Chronological training and validation.
- Validation RMSE comparison.
- Selection of the best-performing candidate.
- Final evaluation on an unseen chronological test set.
- Comparison against a seven-day seasonal baseline.

**Evaluation Metrics**

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error |
| RMSE | Penalizes larger prediction errors |
| R² Score | Measures explained variance |
| Baseline MAE | Compares model performance with a simple forecast |

The latest 20% of observations are reserved for holdout testing.

The trained model and evaluation information are saved using Joblib.

**Generated Artifacts:**
- `sales_model.pkl`
- `features.pkl`
- `model_metrics.pkl`

### 5. Future Sales Prediction

File: `predict.py`

The selected machine learning model generates a recursive forecast for the next 365 days.

Key functionality:
- Uses historical sales information.
- Generates future calendar features.
- Updates lag and rolling features for each forecasted day.
- Predicts sales for future dates.
- Uses previous predictions as inputs for subsequent forecasting steps.

**Output:** `datasets/future_sales_prediction.csv`

### 6. Data Visualization

File: `visualization.py`

Visualizations are created to understand sales performance and business patterns.

Charts include:
- Product category-wise sales.
- Daily sales trends.
- Seven-day rolling average.
- Discount versus sales analysis.
- Order quantity distribution.

## 📊 Interactive Streamlit Dashboard

File: `app.py`

The project includes a business intelligence dashboard called **PredictNextGen – Sales Intelligence**.

The dashboard provides six major sections:

| Dashboard Section | Description |
|---|---|
| Executive Overview | Sales KPIs, orders, average order value and ratings |
| Sales EDA | Explore historical sales trends and patterns |
| Forecast | Visualize future sales predictions |
| Model & Method | Model evaluation, metrics and feature importance |
| Pipeline & Data Quality | Monitor data quality and pipeline artifacts |
| Live Prediction & Accuracy | Work with uploaded sales data and monitor predictions |

### Dashboard Highlights

- Interactive date-range filtering.
- Product category and regional filtering.
- Historical sales visualization.
- Business performance indicators.
- Future sales forecasting.
- Model performance monitoring.
- Feature importance analysis.
- CSV upload and download functionality.
- Live prediction and accuracy tracking.

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/atulverma-0312/Ecommerce-Sales-Prediction-.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Ecommerce-Sales-Prediction-
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Data Science Pipeline

Execute the scripts in the following order:

```bash
python data_cleaning.py
```

```bash
python eda.py
```

```bash
python feature_engineering.py
```

```bash
python train_model.py
```

```bash
python predict.py
```

### Step 6: Run the Interactive Dashboard

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 📈 Business Applications

This project demonstrates how machine learning and data analytics can support:

- Sales forecasting and revenue planning.
- Inventory and stock management.
- Product performance analysis.
- Regional sales comparisons.
- Business performance monitoring.
- Historical trend analysis.
- Data-driven decision-making.

## ⚠️ Limitations

- Forecasting performance depends on the quality and quantity of historical data.
- Long-term recursive forecasts can accumulate prediction errors.
- Future promotions, holidays, market changes, and external factors are not explicitly modeled.
- Future order counts and units sold are estimated using previous weekday patterns.
- Model performance should be evaluated on new business data before operational use.

The forecast should be treated as an estimate rather than a guaranteed sales outcome.

## 🚀 Future Improvements

- Incorporate holiday and promotional event features.
- Experiment with XGBoost and other forecasting algorithms.
- Add advanced time-series models.
- Improve forecasting through hyperparameter optimization.
- Integrate automated model retraining.
- Add real-time data integration.
- Deploy the application to Streamlit Community Cloud.
- Introduce more advanced business intelligence visualizations.

## 👨‍💻 Author

**Atul Verma**

B.Tech – Computer Science (Data Science & Artificial Intelligence)

Interested in:
- Data Analytics
- Data Science
- Machine Learning
- Business Intelligence
- Generative AI

**GitHub:** [atulverma-0312](https://github.com/atulverma-0312)

**Project Repository:** [E-Commerce Sales Prediction](https://github.com/atulverma-0312/Ecommerce-Sales-Prediction-)

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

Your feedback and suggestions are always welcome!

---

<p align="center">
  <b>Turning Data into Insights, and Insights into Better Decisions.</b>
</p>

