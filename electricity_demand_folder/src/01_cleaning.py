import pandas as pd
import numpy as np
import os

df = pd.read_excel("data/mentah/dataset_raw.xlsx")

target = "electricity_demand"

df = df.dropna(subset=[target])

hapus_kolom = [col for col in df.columns 
                if ("electricity" in col.lower() or "generation" in col.lower()) 
                and col != target]

df = df.drop(columns=hapus_kolom)

df = df.select_dtypes(include=[np.number])

os.makedirs("data/hasil_proses", exist_ok=True)

df.to_csv("data/hasil_proses/clean_data.csv", index=False)

print("Shape:", df.shape)
