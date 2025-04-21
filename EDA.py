import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned dataset
df = pd.read_csv("C:/Users/Asus/OneDrive/Documents/GitHub/PRT564_Assignment2/Final_data_avino_weather_filled.csv")

# Set visual style
sns.set(style="whitegrid")

# -------------------------
# 1. Distribution of Bird Positives
# -------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["Total number of birds testing positive"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Birds Testing Positive")
plt.xlabel("Total Number of Birds Testing Positive")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# -------------------------
# 2. Boxplots: Weather vs Bird Positives
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(y="AvgTemp", x="Total number of birds testing positive", data=df, ax=axes[0])
axes[0].set_title("AvgTemp vs Bird Positives")
axes[0].set_xlabel("Bird Positives")
axes[0].set_ylabel("AvgTemp (°C)")

sns.boxplot(y="AvgWindSpeed", x="Total number of birds testing positive", data=df, ax=axes[1])
axes[1].set_title("AvgWindSpeed vs Bird Positives")
axes[1].set_xlabel("Bird Positives")
axes[1].set_ylabel("Wind Speed (km/h)")

sns.boxplot(y="AvgPressure", x="Total number of birds testing positive", data=df, ax=axes[2])
axes[2].set_title("AvgPressure vs Bird Positives")
axes[2].set_xlabel("Bird Positives")
axes[2].set_ylabel("Pressure (mbar)")

plt.tight_layout()
plt.show()

# -------------------------
# 3. Correlation Heatmap
# -------------------------
plt.figure(figsize=(8, 6))
corr = df[["AvgTemp", "AvgWindSpeed", "AvgPressure", "Latitude", "Longitude", "Total number of birds testing positive"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
