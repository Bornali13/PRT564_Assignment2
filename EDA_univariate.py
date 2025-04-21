import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("C:/Users/Asus/OneDrive/Documents/GitHub/PRT564_Assignment2/Final_data_cleaned.csv")

# Set visual style
sns.set(style="whitegrid")

# List of numeric columns to plot
columns = [
    "AvgTemp",
    "AvgWindSpeed",
    "AvgPressure",
    "Latitude",
    "Longitude",
    "Total number of birds testing positive"
]

# Create a 2x3 subplot layout
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Loop through columns and create histograms
for i, col in enumerate(columns):
    ax = axes[i // 3, i % 3]
    sns.histplot(df[col], kde=True, ax=ax, color="cornflowerblue")
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

# Adjust layout
plt.tight_layout()
plt.show()
