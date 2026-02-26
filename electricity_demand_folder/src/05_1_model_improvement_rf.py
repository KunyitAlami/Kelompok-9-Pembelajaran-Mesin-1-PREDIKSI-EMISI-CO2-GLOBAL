import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

preprocessor = joblib.load("data/hasil_proses/preprocessor.pkl")
X_train, X_test, y_train, y_test = joblib.load("data/hasil_proses/split_data.pkl")

X_train_prep = preprocessor.transform(X_train)
X_test_prep = preprocessor.transform(X_test)

depths = [5, 10, 20, None]

results_list = []

for depth in depths:
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=depth,
        oob_score=True,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_prep, y_train)
    
    y_pred = rf.predict(X_test_prep)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results_list.append([
        f"RandomForest_depth_{depth}",
        mae,
        rmse,
        r2,
        rf.oob_score_
    ])

results_df = pd.DataFrame(
    results_list,
    columns=["model", "MAE", "RMSE", "R2", "OOB_Score"]
)

best_model_index = results_df["R2"].idxmax()
best_depth = depths[best_model_index]

rf_best = RandomForestRegressor(
    n_estimators=200,
    max_depth=best_depth,
    oob_score=True,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

rf_best.fit(X_train_prep, y_train)

importances = rf_best.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

top10 = importance_df.head(10)

os.makedirs("results", exist_ok=True)
top10.to_csv("results/rf_top10_feature_importance.csv", index=False)

existing = pd.read_csv("results/eksperimen_log.csv")
updated = pd.concat([existing, results_df], ignore_index=True)
updated.to_csv("results/eksperimen_log.csv", index=False)

print(results_df)
print("\nTop 10 Feature Importance:")
print(top10)
print("\nRandom Forest tuning selesai")