import random

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Make random operations repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str = "configs/default.yaml") -> dict:
    """Read a YAML config file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)