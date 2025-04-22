import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("Final_data_avino_weather_filled.csv")

# ----- 1. Overview -----
print("\n✅ Dataset Shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n✅ Column Data Types:")
print(df.dtypes)

# ----- 2. Missing Values -----
print("\n❗ Missing Values Per Column:")
print(df.isnull().sum())

# ----- 3. Summary Statistics -----
print("\n📊 Summary Statistics:")
print(df.describe())

