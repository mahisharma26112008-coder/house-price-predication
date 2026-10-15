# house-price-predication

1. Introduction:
House Price Prediction is a Machine Learning project that predicts the price of a house based on features such as area, number of bedrooms, location, and other property details.
2. Objective:
To build a machine learning model that estimates house prices using historical housing data.
3. Technologies Used:
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
4. Algorithm: Linear Regression
5. Dataset: Housing dataset containing house areas, bedrooms, and prices.
6. Applications:
- Real estate price estimation
- Property investment analysis
- Housing market analysis
7. Conclusion:
This project demonstrates how machine learning can use historical data to predict house prices

"""
House Price Prediction
Uses the California Housing dataset (built into scikit-learn) and compares
Linear Regression with Random Forest.

Install:  pip install pandas numpy scikit-learn matplotlib joblib
Run:      python house_price_prediction.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data -----------------------------------------------------------
# To use your own CSV instead:
#   df = pd.read_csv("house_data.csv")
#   X = df.drop("price", axis=1); y = df["price"]
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target  # target = median house value (in $100,000s)
print("Shape:", X.shape)
print(X.head(), "\n")

# 2. Train / test split --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Models --------------------------------------------------------------
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)),
    ]),
}

# 4. Train and evaluate --------------------------------------------------
results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {
        "pipe": pipe,
        "pred": pred,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    }
    print(f"{name}: MAE={results[name]['MAE']:.3f}  "
          f"RMSE={results[name]['RMSE']:.3f}  R2={results[name]['R2']:.3f}")

# 5. Pick best model -----------------------------------------------------
best_name = max(results, key=lambda k: results[k]["R2"])
best = results[best_name]
print(f"\nBest model: {best_name}")

# 6. Plots ---------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best["pred"], alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual price (x $100k)")
plt.ylabel("Predicted price (x $100k)")
plt.title(f"Actual vs Predicted - {best_name}")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")

if best_name == "Random Forest":
    importances = pd.Series(
        best["pipe"].named_steps["model"].feature_importances_, index=X.columns
    ).sort_values()
    plt.figure(figsize=(7, 4))
    importances.plot(kind="barh")
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")

# 7. Save model and predict on a new house -------------------------------
joblib.dump(best["pipe"], "house_price_model.joblib")

new_house = pd.DataFrame([{
    "MedInc": 5.0,        # median income (x $10k)
    "HouseAge": 20,
    "AveRooms": 6.0,
    "AveBedrms": 1.0,
    "Population": 1500,
    "AveOccup": 3.0,
    "Latitude": 34.05,
    "Longitude": -118.25,
}])
price = best["pipe"].predict(new_house)[0] * 100_000
print(f"\nPredicted price for the new house: ${price:,.0f}")

plt.show()
