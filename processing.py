from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # For saving the trained model
from datetime import datetime
import pandas as pd
import numpy as np

current_year = datetime.now().year
# Create the annotation text with the copyright symbol
copyright_text = f"<b>© {current_year} NEB Synergy</b>"

# Load the data
df = pd.read_excel("./updated_stats_file.xlsx")  # Replace with your file path
df = df.dropna()

# Assuming df is your dataframe
X = df[["Altitude m", "Average Temperature C", "Average Humidity"]]
y = df["Trailing twelve-month TTM PUE"]
df = df.dropna()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred_precursor = model.predict(X_test)
y_pred = np.round(y_pred_precursor, 2)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
residuals = y_test - y_pred
# Calculate the standard deviation of the residuals
std_dev_residuals = np.std(residuals)
print(f"Standard Deviation of Residuals (Actual - Predicted PUE): {std_dev_residuals}")

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the trained model
joblib.dump(model, "ttm_pue_model.pkl")
print("Model saved as ttm_pue_model.pkl")