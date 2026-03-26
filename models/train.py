import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train(pipeline, train_scaled):
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    X_train = train_scaled.drop("Transported", axis = 1)
    y_train = train_scaled["Transported"]

    with mlflow.start_run() as run:

        pipeline.fit(X_train, y_train)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", 0.03)
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("penalty", "l2")
        mlflow.log_param("max_iter", 491)

        mlflow.sklearn.log_model(pipeline, "model")

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(pipeline, "artifacts/model.pkl")

        print("Model trained")
        return run.info.run_id

if __name__ == "__main__":
    train(None)