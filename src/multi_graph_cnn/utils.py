"""Utils functions for the project"""

from box import Box
from typing import Literal
import numpy as np
import yaml
import logging

import torch


def load_config(config_path="config.yaml", return_type: Literal["dict", "box"] = "box"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if return_type == "dict":
        return config
    elif return_type == "box":
        return Box(config)
    else:
        raise ValueError("return_type not supported, use either 'dict' or 'box'")


def get_logger(name: str = "main", level: int = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def extract_nested_dict(d: dict, type_element=None):
    # Basically check if d is a dict
    # Needed for HDF5-format
    if not hasattr(d, "keys"):
        if type_element is None:
            return d
        return type_element(d)  # cast to type_element if provided

    # Then we can iterate on keys of d
    res = {}
    for key in d.keys():
        res[key] = extract_nested_dict(d[key], type_element)

    return res

def sparse_mx_to_torch(sparse_mx):
    return torch.tensor(sparse_mx.toarray(), dtype=torch.float32)

