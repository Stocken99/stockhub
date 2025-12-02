# test_preprocessing.py
from src.preprocessing.preprocess_data import preprocess
import yaml
from pathlib import Path
import pandas as pd

# Temporary config
cfg_path = "config/preprocess.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# Use temporary output path
tmp_output = Path("processed_temp.csv")
cfg["output_dataset"] = str(tmp_output)

# Save temporary config
tmp_config = Path("config_temp.yaml")
with open(tmp_config, "w") as f:
    yaml.dump(cfg, f)

# Run preprocessing
output_path = preprocess(config_path=str(tmp_config))

# Check results
df = pd.read_csv(output_path)
assert not df.empty
assert "date" in df.columns

print("Preprocessing test passed!")

