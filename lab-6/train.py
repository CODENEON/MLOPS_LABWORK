import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the housing dataset
data = pd.read_csv('housing.csv')
X, y = data.drop('PRICE', axis=1), data['PRICE']
X = X.astype({col: 'float64' for col in X.select_dtypes(include=['int']).columns})

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate a model
def train_and_evaluate_model(model, model_name, params):
    # Fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print model parameters and metrics
    print(f"{model_name} Parameters: {params}")
    print(f"{model_name} MSE: {mse}")
    print(f"{model_name} R² Score: {r2}")

    return mse, r2

# Train and evaluate Linear Regression
linear_model = LinearRegression()
linear_params = {
    "fit_intercept": linear_model.fit_intercept,
    "normalize": False,  # normalize is deprecated in new versions of sklearn, but included if needed
}
linear_mse, linear_r2 = train_and_evaluate_model(linear_model, "Linear Regression", linear_params)

# Train and evaluate Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_params = {
    "n_estimators": rf_model.n_estimators,
    "max_depth": rf_model.max_depth,
    "random_state": rf_model.random_state,
}
rf_mse, rf_r2 = train_and_evaluate_model(rf_model, "Random Forest", rf_params)

# Compare the models and select the best one
if rf_mse < linear_mse:
    best_model_name = "Random Forest"
    best_mse = rf_mse
else:
    best_model_name = "Linear Regression"
    best_mse = linear_mse

print(f"\nBest model: {best_model_name} with MSE: {best_mse}")

