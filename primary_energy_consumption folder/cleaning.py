import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

input_path = "dataset_after_preprocessing.csv"
df = pd.read_csv(input_path)

target = "primary_energy_consumption"

if target not in df.columns:
    raise ValueError(f"Target '{target}' tidak ditemukan di dataset")

df = df.dropna(subset=[target])

X = df.drop(columns=[target])
y = df[target]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_cols]

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

q1 = y.quantile(0.25)
q3 = y.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

is_outlier = ((y < lower) | (y > upper)).astype(int)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index)

df_out = X_scaled.copy()
df_out[target] = y.loc[X_scaled.index].values
df_out["is_outlier"] = is_outlier.loc[X_scaled.index].values

print("Input shape:", df.shape)
print("Output shape:", df_out.shape)
print("Jumlah fitur:", X_scaled.shape[1])
print("Jumlah data:", df_out.shape[0])
print("Jumlah outlier rows:", int(df_out["is_outlier"].sum()))

df_out.to_csv("dataset_clean_scaled.csv", index=False)