import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (replace with your actual path or DataFrame)
df = pd.read_csv('Final_data_robust_scaled.csv')

# Optional: Drop non-numeric or irrelevant columns if needed
df = df.drop(columns=['Week number','Start Date','Location (county)'])

# Compute correlation matrix
correlation_matrix = df.corr()

# Display correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()