import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

# Load the housing dataset
data = pd.read_csv('housing.csv')
X, y = data.drop('PRICE', axis=1), data['PRICE']
X = X.astype({col: 'float64' for col in X.select_dtypes(include=['int']).columns})

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Function to train and log a model
def train_and_log_model(model, model_name, params):
    with mlflow.start_run():
        # Fit the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log model parameters and metrics
        mlflow.log_params(params)  # Log the model parameters
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2_score", r2)

        # Set a tag to describe the run
        mlflow.set_tag("Training Info", f"Basic {model_name} model for housing data")

        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"{model_name}_model",
            signature=signature,
            input_example=X_train
        )

        print(f"{model_name} MSE: {mse}")
        print(f"{model_name} R2_score: {r2}")
        
        return model, mse, r2

# Train and log Linear Regression
linear_model = LinearRegression()
linear_params = {
    "fit_intercept": linear_model.fit_intercept,
    "normalize": False,  # normalize is deprecated in new versions of sklearn, but included if needed
}
linear_regression_model, linear_mse, linear_r2 = train_and_log_model(linear_model, "Linear_Regression", linear_params)

# Train and log Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_params = {
    "n_estimators": rf_model.n_estimators,
    "max_depth": rf_model.max_depth,
    "random_state": rf_model.random_state,
}
random_forest_model, rf_mse, rf_r2 = train_and_log_model(rf_model, "Random_Forest", rf_params)

# Compare the models and select the best one
if rf_mse < linear_mse:
    best_model = random_forest_model
    best_model_name = "Random_Forest"
    best_mse = rf_mse
else:
    best_model = linear_regression_model
    best_model_name = "Linear_Regression"
    best_mse = linear_mse

print(f"\nBest model: {best_model_name} with MSE: {best_mse}")

# Log and save the best model in the Model Registry
with mlflow.start_run():
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/{best_model_name}_model"
    mlflow.register_model(model_uri, f"Best_Housing_Model: {best_model_name}")
    mlflow.log_metric("MSE", best_mse)
    print(f"{best_model_name} registered in the Model Registry.")
