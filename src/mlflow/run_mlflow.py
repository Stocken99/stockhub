

import mlflow
from mlflow.pyfunc import PythonModel
from src.models.train_chronos import load_config, train_chronos_model


# MLflow PyFunc wrapper
class ChronosMLflowWrapper(PythonModel):
    def load_context(self, context):
        import autogluon.timeseries as agts
        self.model = agts.TimeSeriesPredictor.load(
            context.artifacts["chronos_model"]
        )

    def predict(self, context, model_input):
        return self.model.predict(model_input)


def train_chronos_with_mlflow(processed_csv_path=None, config_path="config/train_chronos.yaml"):
    """
    Orchestrates MLflow tracking around the pure training function.
    """

    cfg = load_config(config_path)

    # Configure MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run() as run:

        # Log experiment parameters
        mlflow.log_params(cfg["params"])

        # ---- Train model (NO mlflow inside this function) ----
        metrics, prediction_file, model_dir = train_chronos_model(
            cfg, processed_csv_path
        )

        # ---- Log metrics ----
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # ---- Log predictions CSV ----
        mlflow.log_artifact(prediction_file)

        # ---- Register model ----
        mlflow_model_path = "chronos_model"

        mlflow.pyfunc.log_model(
            artifact_path=mlflow_model_path,
            python_model=ChronosMLflowWrapper(),
            artifacts={"chronos_model": model_dir},
        )

        model_uri = f"runs:/{run.info.run_id}/{mlflow_model_path}"
        mlflow.register_model(model_uri, "chronos_model")

        print("Model registered as 'chronos_model'")

        return metrics, prediction_file


if __name__ == "__main__":
    train_chronos_with_mlflow()
