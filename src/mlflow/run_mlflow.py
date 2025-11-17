import mlflow
import mlflow.tensorflow
import mlflow.pytorch

MLFLOW_URI ="http://127.0.0.1:5000/"
EXPERIMENT_NAME = "Models_Experiment"
REGISTERED_MODEL_NAME = ""
MODEL = model
PARAMS = params


mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run():
    mlflow.log_params(PARAMS)
    mlflow.log_metrics(
        {
            ## add the metrics 
        },
    )
    register_model_name= REGISTERED_MODEL_NAME 
    mlflow.tensorflow.log_model(MODEL, registered_model_name=REGISTERED_MODEL_NAME) # change the flavor module (for example mlflow.tensorflow, mlflow.sklearn, mlflow.pytorch)