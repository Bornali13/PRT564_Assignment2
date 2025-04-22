import pandas as pd

# Load your dataset
df = pd.read_csv("Final_data_cleaned.csv")

# List of numerical columns to apply IQR-based outlier treatment
columns_to_treat = [
    "AvgTemp",
    "AvgWindSpeed",
    "AvgPressure",
    "Latitude",
    "Longitude",
    "Total number of birds testing positive"
]

# Function to cap outliers using IQR
def treat_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

# Apply to all selected columns
for col in columns_to_treat:
    df = treat_outliers_iqr(df, col)

# Save cleaned dataset
df.to_csv("Final_data_outliers_treated.csv", index=False)
print("✅ Outliers treated and data saved as 'Final_data_outliers_treated.csv'")
