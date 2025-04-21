import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load original and scaled data
df_original = pd.read_csv("Final_data_outliers_treated.csv")
df_scaled = pd.read_csv("Final_data_robust_scaled.csv") 
# Features to compare
features = ["AvgTemp", "AvgWindSpeed", "AvgPressure", "Latitude", "Longitude"]

# Create boxplots before and after scaling
fig, axes = plt.subplots(len(features), 2, figsize=(12, 16))
sns.set(style="whitegrid")

for i, col in enumerate(features):
    sns.boxplot(y=df_original[col], ax=axes[i, 0], color="lightblue")
    axes[i, 0].set_title(f"Before Scaling: {col}", fontsize=10)

    sns.boxplot(y=df_scaled[col], ax=axes[i, 1], color="lightgreen")
    axes[i, 1].set_title(f"After Scaling: {col}", fontsize=10)

plt.tight_layout()
plt.show()
