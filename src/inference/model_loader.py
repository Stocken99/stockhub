import mlflow

def load_model():
    print("Loading Chronos model from MLflow...")
    mlflow.set_tracking_uri("http://localhost:5000")
    model = mlflow.pyfunc.load_model("models:/chronos_model/latest")
    return model
