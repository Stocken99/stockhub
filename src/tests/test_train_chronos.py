# test_train_chronos_plain.py
import pandas as pd
from src.models.train_chronos import prepare_data
from pathlib import Path

def main():
    # Create a temporary CSV file
    tmp_path = Path("tmp_test_dir")
    tmp_path.mkdir(exist_ok=True)
    csv_path = tmp_path / "dummy.csv"

    # Dummy dataset
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10),
        "4. close": range(10)
    })
    df.to_csv(csv_path, index=False)

    # Call prepare_data
    train_df, test_df = prepare_data(str(csv_path), prediction_length=3)

    # Assertions
    assert len(train_df) == 7, f"Expected train length 7, got {len(train_df)}"
    assert len(test_df) == 3, f"Expected test length 3, got {len(test_df)}"
    assert "timestamp" in train_df.columns, "'timestamp' column missing in train_df"
    assert "target" in train_df.columns, "'target' column missing in train_df"

    print("train_chronos prepare_data test passed!")

    # Cleanup
    csv_path.unlink()
    tmp_path.rmdir()

if __name__ == "__main__":
    main()
