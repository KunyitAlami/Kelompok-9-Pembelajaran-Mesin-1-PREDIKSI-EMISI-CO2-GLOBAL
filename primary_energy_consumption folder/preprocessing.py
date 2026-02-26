import pandas as pd
import numpy as np

file_path = "dataset fix (raw).xlsx"
df = pd.read_excel(file_path)

print("Shape awal:", df.shape)

df.drop(columns=["country", "iso_code"], errors="ignore", inplace=True)

target = "primary_energy_consumption"

if target not in df.columns:
    raise ValueError(f"Target '{target}' tidak ditemukan di dataset")

columns_to_drop = [
    col for col in df.columns
    if ("_per_capita" in col)
    or ("_share" in col)
    or ("_change" in col)
]

df.drop(columns=columns_to_drop, inplace=True)

missing_percentage = df.isnull().mean() * 100
high_missing_cols = missing_percentage[missing_percentage > 70].index
df.drop(columns=high_missing_cols, inplace=True)

constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
df.drop(columns=constant_cols, inplace=True)

df = df.dropna(subset=[target])

X = df.drop(columns=[target])
y = df[target]

print("Shape akhir:", df.shape)
print("Jumlah fitur:", X.shape[1])
print("Jumlah data:", X.shape[0])

df.to_csv("dataset_after_preprocessing.csv", index=False)