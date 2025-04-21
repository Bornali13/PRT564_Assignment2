import pandas as pd

# Load your dataset
df = pd.read_csv("C:/Users/Asus/OneDrive/Documents/GitHub/PRT564_Assignment2/Final_data_avino_weather_filled.csv")

# -----------------------------
# 🔍 Step 1: Check for missing values
# -----------------------------
missing_summary = df.isna().sum()
print("Missing values per column:\n", missing_summary)

# -----------------------------
# 🧹 Step 2: Handle missing values
# Option A: Fill weather-related columns with column mean
weather_cols = ["AvgTemp", "AvgPressure", "AvgWindSpeed"]
df[weather_cols] = df[weather_cols].fillna(df[weather_cols].mean())

# Option B: Drop rows with missing Latitude/Longitude (usually critical)
df = df.dropna(subset=["Latitude", "Longitude"])

# -----------------------------
# ✅ Step 3: Confirm no missing values
# -----------------------------
print("\nRemaining missing values:\n", df.isna().sum())

# Save the cleaned version if needed
df.to_csv("Final_data_cleaned.csv", index=False)
