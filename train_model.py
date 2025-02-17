import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Data (Replace 'example.csv' with your actual dataset)
data = pd.read_csv("example.csv")  # Ensure this file is in the same directory

# Step 2: Preprocessing
data.fillna(method='ffill', inplace=True)

# Selecting Features and Target
features = data[['temperature', 'charge_cycles', 'discharge_rate']]
target = data['battery_life']

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 3: Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate Model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model Trained! MSE: {mse}, R2: {r2}")

# Step 5: Save Model using Joblib
joblib.dump(model, "rf_battery_life_model.pkl")
print("Model saved as 'rf_battery_life_model.pkl'")
