import mlflow
from src.preprocessing.preprocess_data import preprocess
from src.mlflow.run_mlflow import train_chronos_with_mlflow


def main():
    print("Running Pipeline...")

    # 1. Preprocess
    print("Preprocessing data...")
    processed_path = preprocess()
    print(f"Preprocessed dataset saved to: {processed_path}")

    # 2. Training + MLflow
    print("Connecting to MLflow...")
    metrics, artifact_path = train_chronos_with_mlflow(processed_path)

    print("Pipeline Finished!")
    print("Metrics:", metrics)
    print("Predictions saved to:", artifact_path)


if __name__ == "__main__":
    main()

