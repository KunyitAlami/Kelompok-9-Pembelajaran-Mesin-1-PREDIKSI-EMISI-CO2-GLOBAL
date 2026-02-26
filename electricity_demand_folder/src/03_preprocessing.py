import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

df = pd.read_csv("data/hasil_proses/clean_data.csv")

target = "electricity_demand"

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor.fit(X_train)

os.makedirs("data/hasil_proses", exist_ok=True)

joblib.dump(preprocessor, "data/hasil_proses/preprocessor.pkl")
joblib.dump((X_train, X_test, y_train, y_test), "data/hasil_proses/split_data.pkl")

print("Preprocessing selesai")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)