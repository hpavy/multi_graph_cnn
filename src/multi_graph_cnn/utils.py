"""Utils functions for the project"""

from box import Box
from typing import Literal
import numpy as np
import yaml
import logging

from torch.utils.tensorboard import SummaryWriter # <--- Import this
import os

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


def get_tensorboard_writer(config):
    """Creates a TensorBoard SummaryWriter in the output directory"""
    # Saves logs to: saved_models/YYYYMMDD-HHMMSS/runs/
    log_dir = os.path.join(config.output_dir, config.tensorboard_dir)
    return SummaryWriter(log_dir=log_dir)


def get_svd_initialization(sparse_matrix, rank, device):
    """
    Performs SVD on the sparse rating matrix to initialize W and H.
    Args:
        sparse_matrix (torch.Tensor): The input matrix M (MxN).
        rank (int): The rank 'r' for factorization.
    Returns:
        W (MxR), H (NxR)
    """
    # 1. Ensure matrix is on the correct device
    matrix =  torch.tensor(sparse_matrix, dtype=torch.float32).to(device)

    # 2. Perform Randomized SVD (Fast for large matrices)
    # U: (M, r), S: (r,), V: (N, r)
    U, S, V = torch.svd_lowrank(matrix, q=rank, niter=2)

    # 3. Distribute Sigma to balance the factors (Symmetric initialization)
    # W = U * sqrt(S)
    # H = V * sqrt(S)
    # So that W @ H.T approx M
    sqrt_S = torch.diag(torch.sqrt(S))
    
    W_init = torch.matmul(U, sqrt_S)
    H_init = torch.matmul(V, sqrt_S)

    return W_init, H_init
