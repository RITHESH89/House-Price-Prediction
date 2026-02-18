import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from preprocessing import preprocess_data

# Load data
X, y = preprocess_data("data/train.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost model
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Convert back from log scale
predictions = np.expm1(predictions)
y_test_original = np.expm1(y_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
print("RMSE:", rmse)

# Save model
joblib.dump(model, "model/model.pkl")
