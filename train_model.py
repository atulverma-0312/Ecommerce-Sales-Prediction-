from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load the daily sales data and keep it in date order.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"

data = pd.read_csv(
    DATA_DIR / "feature_engineered.csv",
    index_col="Order_Date",
    parse_dates=True,
).sort_index()

# 2. Separate the answer (Sales) from the information used to predict it.
features = [column for column in data.columns if column != "Sales"]
X = data[features]
y = data["Sales"]

# 3. Use older dates for training and newer dates for testing.
#    Do not shuffle time-series data: the test period should come after training.
split_index = int(len(data) * 0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 4. Select a model using a chronological validation split inside the training
#    period. The final holdout remains untouched until evaluation.
validation_index = int(len(X_train) * 0.8)
X_fit = X_train.iloc[:validation_index]
X_validation = X_train.iloc[validation_index:]
y_fit = y_train.iloc[:validation_index]
y_validation = y_train.iloc[validation_index:]

if X_fit.empty or X_validation.empty or X_test.empty:
    raise ValueError("Not enough dated rows for training, validation, and test splits.")

random_forest = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
candidates = {
    "RandomForestRegressor": random_forest,
    "LogTargetRandomForest": TransformedTargetRegressor(
        regressor=RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    ),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    ),
}

validation_rmse = {}
for name, candidate in candidates.items():
    candidate.fit(X_fit, y_fit)
    validation_predictions = np.maximum(0, candidate.predict(X_validation))
    validation_rmse[name] = mean_squared_error(
        y_validation, validation_predictions
    ) ** 0.5

selected_name = min(validation_rmse, key=validation_rmse.get)
model = candidates[selected_name]
model.fit(X_train, y_train)

# 5. Check predictions on dates the model did not train on.
predictions = np.maximum(0, model.predict(X_test))

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print(f"Selected model: {selected_name}")
print("Holdout results (latest 20% of dates)")
print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R2:   {r2:.4f}")
print("\nValidation RMSE by model")
for name, score in sorted(validation_rmse.items(), key=lambda item: item[1]):
    print(f"{name}: {score:,.2f}")

# Compare with a simple forecast: use sales from the same day last week.
baseline_predictions = X_test["Lag_7"].clip(lower=0)
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_r2 = r2_score(y_test, baseline_predictions)
print(f"\n7-day baseline MAE: {baseline_mae:,.2f}")
print(f"7-day baseline R2:  {baseline_r2:.4f}")
if mae < baseline_mae:
    print(f"{selected_name} has a lower MAE than the baseline.")
else:
    print(f"{selected_name} did not beat the baseline; consider improving the model.")

# Show feature importance when the selected estimator exposes it.
if hasattr(model, "feature_importances_"):
    importance = pd.Series(
        model.feature_importances_,
        index=features,
    ).sort_values(ascending=False)
    print("\nTop 10 important features")
    print(importance.head(10).to_string())

# Save holdout metrics so the dashboard can report measured test performance.
evaluation = {
    "model": selected_name,
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "baseline_mae": baseline_mae,
    "baseline_r2": baseline_r2,
    "validation_rmse": validation_rmse,
    "train_end": X_train.index[-1].isoformat(),
    "test_start": X_test.index[0].isoformat(),
    "test_end": X_test.index[-1].isoformat(),
}

# 6. Retrain on all available data, then save for later use.
model.fit(X, y)

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODEL_DIR / "sales_model.pkl")
joblib.dump(features, MODEL_DIR / "features.pkl")
joblib.dump(evaluation, MODEL_DIR / "model_metrics.pkl")

print("Model and feature list saved in the models folder.")
