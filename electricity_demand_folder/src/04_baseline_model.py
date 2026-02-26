import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

preprocessor = joblib.load("data/hasil_proses/preprocessor.pkl")
X_train, X_test, y_train, y_test = joblib.load("data/hasil_proses/split_data.pkl")

X_train_prep = preprocessor.transform(X_train)
X_test_prep = preprocessor.transform(X_test)

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train_prep, y_train)
y_pred_dummy = dummy.predict(X_test_prep)

mae_dummy = mean_absolute_error(y_test, y_pred_dummy)
rmse_dummy = np.sqrt(mean_squared_error(y_test, y_pred_dummy))
r2_dummy = r2_score(y_test, y_pred_dummy)

lr = LinearRegression()
lr.fit(X_train_prep, y_train)
y_pred_lr = lr.predict(X_test_prep)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

results = pd.DataFrame([
    ["DummyRegressor", mae_dummy, rmse_dummy, r2_dummy],
    ["LinearRegression", mae_lr, rmse_lr, r2_lr]
], columns=["model", "MAE", "RMSE", "R2"])

os.makedirs("results", exist_ok=True)
results.to_csv("results/eksperimen_log.csv", index=False)

print(results)
