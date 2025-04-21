import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the robust-scaled dataset
df = pd.read_csv("Final_data_robust_scaled.csv")

# Define predictors and target
X = df[["AvgTemp", "AvgWindSpeed", "AvgPressure", "Latitude", "Longitude"]]
y = df["Total number of birds testing positive"]

# Add intercept
X_const = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X_const).fit()

# Print regression summary
print(model.summary())

# Predict values
y_pred = model.predict(X_const)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

# Print metrics
print(f"\n✅ Model Performance Metrics:")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
