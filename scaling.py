import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Load the IQR-treated dataset
df = pd.read_csv("Final_data_outliers_treated.csv")

# Select numerical columns to scale
features_to_scale = [
    "AvgTemp",
    "AvgWindSpeed",
    "AvgPressure",
    "Latitude",
    "Longitude"
]


#  RobustScaler (robust to outliers)
robust_scaler = RobustScaler()
df_robust_scaled = df.copy()
df_robust_scaled[features_to_scale] = robust_scaler.fit_transform(df[features_to_scale])

# Savescaled versions
df_robust_scaled.to_csv("Final_data_robust_scaled.csv", index=False)

print("- Final_data_robust_scaled.csv")
