import pandas as pd
import mlflow
import os
from autogluon.timeseries import TimeSeriesPredictor
import yaml

#Config
def load_config(path="config/train_chronos.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

#Data preperation
def prepare_data(csv_path, prediction_length):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    # Chronos needs: timestamp, target, id
    df = df[["date", "4. close"]].rename(columns={"date": "timestamp", "4. close": "target"})
    df["item_id"] = "AAPL"

    df = df.sort_values("timestamp")

    train_df = df.iloc[:-prediction_length]
    test_df = df.iloc[-prediction_length:]

    return train_df, test_df

#Wrapper
class ChronosMLflowWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import autogluon.timeseries as agts
        self.model = agts.TimeSeriesPredictor.load(context.artifacts["chronos_model"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)

#Train
def train_chronos(processed_csv_path=None, config_path="config/train_chronos.yaml"):
    """Runs Chronos training using YAML config + optional CSV override."""
    
    # Load YAML
    cfg = load_config(config_path)
    freq = cfg["params"].get("freq", "1D")

    # OVERRIDE CSV path if pipeline provided one
    if processed_csv_path is not None:
        cfg["data"]["processed_csv"] = processed_csv_path

    # MLflow setup
    mlflow.set_tracking_uri(cfg["mlflow"]["uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run() as run:
        mlflow.log_params(cfg["params"])

        # Load and prepare data
        train_df, test_df = prepare_data(
            cfg["data"]["processed_csv"],
            cfg["params"]["prediction_length"]
        )

        # Create AutoGluon Chronos predictor
        predictor = TimeSeriesPredictor(
            prediction_length=cfg["params"]["prediction_length"],
            target="target",
            eval_metric="RMSE",
            path="models/chronos_model/",
            freq=freq
        )
        

        predictor.fit(
            train_df,
            hyperparameters={"Chronos": {}}
        )

        # Predict on the future horizon
        predictions = predictor.predict(train_df)

        merged = predictions.merge(
            test_df[["timestamp", "target", "item_id"]],
            on=["timestamp", "item_id"],
            how="left"
        )

        merged["sq_error"] = (merged["mean"] - merged["target"]) ** 2
        mse = merged["sq_error"].mean()
        mlflow.log_metric("mse", mse)

        # Save predictions
        out_path = cfg["outputs"]["prediction_csv"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        merged.to_csv(out_path, index=False)
        mlflow.log_artifact(out_path)

        print("Training finished. MSE:", mse)

        #Log model
        mlflow_pyfunc_path = "chronos_mlflow_model"

        mlflow.pyfunc.log_model(
            artifact_path=mlflow_pyfunc_path,
            python_model=ChronosMLflowWrapper(),
            artifacts={"chronos_model": predictor.path},
        )

        #Register model
        model_uri = f"runs:/{run.info.run_id}/{mlflow_pyfunc_path}"
        mlflow.register_model(model_uri, "chronos_model")

        print("Model registered as 'chronos_model'")

        return {"mse": mse}, out_path


if __name__ == "__main__":
    train_chronos()


