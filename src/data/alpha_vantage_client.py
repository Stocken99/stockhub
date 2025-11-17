import yaml
import requests
import os

class AlphaVantageClient:

    def __init__(self, config_path="config/alpha_vantage.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["alpha_vantage"]

        self.base_url = self.config["base_url"]
        self.api_key = os.getenv("ALPHAVANTAGE_API_KEY") or self.config["api_key"]

        if not self.api_key:
            raise ValueError("No API key found. Set ALPHAVANTAGE_API_KEY env variable.")

    def fetch(self, function_name: str, **kwargs):
        if function_name not in self.config["functions"]:
            raise ValueError(f"Unknown function: {function_name}")

        # load base params from yaml
        params = self.config["functions"][function_name]["params"].copy()

        # format {ticker}, {time_from}, {time_to} etc
        for key, val in params.items():
            if isinstance(val, str) and "{" in val:
                params[key] = val.format(**kwargs)

        params["apikey"] = self.api_key

        r = requests.get(self.base_url, params=params)
        data = r.json()

        # optional rate-limit handling
        if "Note" in data:
            print("⚠️ RATE LIMIT:", data["Note"])
        return data


