"""src.config"""
from pathlib import Path
import torch
from dotenv import dotenv_values

env_path = Path(".") / ".env"

config = dotenv_values(env_path)

MODEL_CONFIG = {
    "model_path": config["MODEL_PATH"],
}

API_CONFIG = {
    "API_KEY": config["API_KEY"],
}
