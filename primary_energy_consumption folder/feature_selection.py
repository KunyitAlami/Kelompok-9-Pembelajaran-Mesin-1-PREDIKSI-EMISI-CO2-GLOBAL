import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

input_path = "dataset_clean_scaled.csv"
df = pd.read_csv(input_path)

target = "primary_energy_consumption"

df_model = df.drop(columns=["is_outlier"], errors="ignore")

X = df_model.drop(columns=[target])
y = df_model[target]

correlation = X.corrwith(y).abs()
selected_by_corr = correlation[correlation > 0.20].index.tolist()
X = X[selected_by_corr]

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif_data

while True:
    vif = calculate_vif(X)
    max_vif = vif["VIF"].max()
    if max_vif > 20:
        feature_to_drop = vif.sort_values("VIF", ascending=False)["feature"].iloc[0]
        X = X.drop(columns=[feature_to_drop])
    else:
        break

df_final = X.copy()
df_final[target] = y.loc[X.index].values
df_final["is_outlier"] = df["is_outlier"]

print("Jumlah fitur setelah correlation filter:", len(selected_by_corr))
print("Jumlah fitur final setelah VIF iterative:", X.shape[1])
print("Jumlah data:", df_final.shape[0])

df_final.to_csv("dataset_final_features.csv", index=False)