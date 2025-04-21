import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load original and treated datasets
df_original = pd.read_csv("Final_data_cleaned.csv")
df_treated = pd.read_csv("Final_data_outliers_treated.csv")

# Define the columns to visualize
cols = [
    "AvgTemp",
    "AvgWindSpeed",
    "AvgPressure",
    "Latitude",
    "Longitude",
    "Total number of birds testing positive"
]

# Set visual style
sns.set(style="whitegrid")

# Create a 2-column grid of boxplots (before vs after)
fig, axes = plt.subplots(len(cols), 2, figsize=(10, 16))

for i, col in enumerate(cols):
    # Before treatment
    sns.boxplot(data=df_original, y=col, ax=axes[i, 0], color="salmon")
    axes[i, 0].set_title(f"Before Outlier Treatment: {col}", fontsize=8)

    # After treatment
    sns.boxplot(data=df_treated, y=col, ax=axes[i, 1], color="lightgreen")
    axes[i, 1].set_title(f"After Outlier Treatment: {col}", fontsize=8)

# Layout adjustment
plt.tight_layout()
plt.show()