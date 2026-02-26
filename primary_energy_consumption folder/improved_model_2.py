import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

input_path = "dataset_final_features.csv"
df = pd.read_csv(input_path)

target = "primary_energy_consumption"

X = df.drop(columns=[target, "is_outlier"], errors="ignore")
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=== IMPROVED MODEL 2 (XGBRegressor) ===")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

results = pd.DataFrame({
    "Model": ["XGBRegressor"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})

results.to_csv("improved_results_xgb.csv", index=False)