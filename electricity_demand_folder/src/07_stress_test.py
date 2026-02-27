import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

preprocessor = joblib.load("data/hasil_proses/preprocessor.pkl")
x_train, x_test, y_train, y_test = joblib.load("data/hasil_proses/split_data.pkl")

x_train_prep = preprocessor.transform(x_train)
x_test_prep = preprocessor.transform(x_test)

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(x_train_prep, y_train)
y_prediction = rf.predict(x_test_prep)

mae_before = mean_absolute_error(y_test, y_prediction)
rmse_before = np.sqrt(mean_squared_error(y_test, y_prediction))
r2_before = r2_score(y_test, y_prediction)

residual = y_test - y_prediction

os.makedirs("results/gambar", exist_ok=True)

plt.figure(figsize=(8,5))
sns.histplot(residual, kde=True)
plt.title("Distribusi Residual (Sebelum Outlier Removal)")
plt.savefig("results/gambar/residual_distribution_before.png")
plt.close()

plt.figure(figsize=(8,5))
plt.scatter(y_prediction, residual, alpha=0.3)
plt.axhline(0, color='red')
plt.title("Residual vs Predicted")
plt.savefig("results/gambar/residual_vs_predicted.png")
plt.close()

Q1 = np.percentile(residual, 25)
Q3 = np.percentile(residual, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

mask = (residual >= lower_bound) & (residual <= upper_bound)

x_test_clean = x_test_prep[mask]
y_test_clean = y_test[mask]

y_prediction_clean = rf.predict(x_test_clean)

mae_after = mean_absolute_error(y_test_clean, y_prediction_clean)
rmse_after = np.sqrt(mean_squared_error(y_test_clean, y_prediction_clean))
r2_after = r2_score(y_test_clean, y_prediction_clean)

residual_clean = y_test_clean - y_prediction_clean

plt.figure(figsize=(8,5))
sns.histplot(residual_clean, kde=True)
plt.title("Distribusi Residual (Setelah Outlier Removal)")
plt.savefig("results/gambar/residual_distribution_after.png")
plt.close()

stress_result = pd.DataFrame([
    ["Sebelum Outlier Removal", mae_before, rmse_before, r2_before],
    ["Setelah Outlier Removal", mae_after, rmse_after, r2_after]
], columns=["Condition", "MAE", "RMSE", "R2"])

stress_result.to_csv("results/stress_test_results.csv", index=False)

print(stress_result)
print("Stress test selesai")