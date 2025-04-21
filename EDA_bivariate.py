import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("Final_data_cleaned.csv")

# Set visual style
sns.set(style="whitegrid")

# Define predictors and target variable
predictors = ["AvgTemp", "AvgWindSpeed", "AvgPressure", "Latitude", "Longitude"]
target = "Total number of birds testing positive"

# Create 5 scatter plots in 2x3 layout with smaller font
plt.figure(figsize=(14, 10))

for i, col in enumerate(predictors, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=col, y=target, data=df, alpha=0.5)
    sns.regplot(x=col, y=target, data=df, scatter=False, color='red')
    plt.title(f"{col} vs Bird Positives", fontsize=8)
    plt.xlabel(col, fontsize=7)
    plt.ylabel(target, fontsize=7)

plt.tight_layout()
plt.show()
