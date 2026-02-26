import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("data/hasil_proses/clean_data.csv")

print(df.info())
print(df.describe())

missing = df.isnull().sum()
print(missing[missing > 0])

os.makedirs("results/gambar", exist_ok=True)

plt.figure(figsize=(8,5))
sns.histplot(df["electricity_demand"], kde=True)
plt.title("Distribusi Electricity Demand")
plt.savefig("results/gambar/distribusi_target.png")
plt.close()

plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("results/gambar/correlation_matrix.png")
plt.close()

print("Selesai")
