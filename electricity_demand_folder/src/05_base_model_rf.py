import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

preprocessor = joblib.load("data/hasil_proses/preprocessor.pkl")
X_train, X_test, y_train, y_test = joblib.load("data/hasil_proses/split_data.pkl")

X_train_prep = preprocessor.transform(X_train)
X_test_prep = preprocessor.transform(X_test)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_prep, y_train)
y_pred_rf = rf.predict(X_test_prep)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

new_result = pd.DataFrame([
    ["RandomForest", mae_rf, rmse_rf, r2_rf]
], columns=["model", "MAE", "RMSE", "R2"])

existing = pd.read_csv("results/eksperimen_log.csv")
updated = pd.concat([existing, new_result], ignore_index=True)
updated.to_csv("results/eksperimen_log.csv", index=False)

print(updated)
print("Base Random Forest Regression selesai")