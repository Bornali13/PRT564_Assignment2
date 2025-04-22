import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Load the dataset
df = pd.read_csv("Final_data_robust_scaled.csv")
df = df.dropna()

# Define target and predictors
target = "Total number of birds testing positive"
predictors = ["AvgTemp", "AvgWindSpeed", "AvgPressure", "Latitude", "Longitude"]

# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))

# Store results
regression_metrics = []

for i, predictor in enumerate(predictors, 1):
    plt.subplot(2, 3, i)

    # Scatter plot
    sns.scatterplot(x=df[predictor], y=df[target], alpha=0.5)

    # Fit model
    X = sm.add_constant(df[predictor])
    y = df[target]
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    # Metrics
    coef = model.params[predictor]
    pval = model.pvalues[predictor]
    r2 = model.rsquared
    rmse = sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)

    # Store metrics
    regression_metrics.append({
        "Predictor": predictor,
        "Coef": coef,
        "R-squared": r2,
        "p-value": pval,
        "RMSE": rmse,
        "MAE": mae
    })

    # Plot regression line
    sns.regplot(x=df[predictor], y=df[target], scatter=False, color="red", line_kws={"linewidth": 2})

    # Title with all metrics
    plt.title(
        f"{predictor} vs Bird Positives\n"
        f"Coef: {coef:.3f}, R²: {r2:.3f}, p: {pval:.4f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}",
        fontsize=8
    )
    plt.xlabel(predictor, fontsize=7)
    plt.ylabel("Bird Positives", fontsize=7)

# Show plots
plt.tight_layout()
plt.show()

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(regression_metrics)
print("\nSimple Linear Regression Summary:")
print(metrics_df)


