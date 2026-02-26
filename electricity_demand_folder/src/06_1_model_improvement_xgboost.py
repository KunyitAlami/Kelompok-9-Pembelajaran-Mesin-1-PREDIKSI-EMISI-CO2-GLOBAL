import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

preprocessor = joblib.load("data/hasil_proses/preprocessor.pkl")
X_train, X_test, y_train, y_test = joblib.load("data/hasil_proses/split_data.pkl")

X_train_prep = preprocessor.transform(X_train)
X_test_prep = preprocessor.transform(X_test)

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="rmse"
)

xgb.fit(
    X_train_prep,
    y_train,
    eval_set=[(X_test_prep, y_test)],
    verbose=False
)

y_pred = xgb.predict(X_test_prep)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

result_df = pd.DataFrame([
    ["XGBoost_Modified", mae, rmse, r2]
], columns=["model", "MAE", "RMSE", "R2"])

existing = pd.read_csv("results/eksperimen_log.csv")
updated = pd.concat([existing, result_df], ignore_index=True)
updated.to_csv("results/eksperimen_log.csv", index=False)

print(result_df)
print("\nXGBoost modifikasi selesai")