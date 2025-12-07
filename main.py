"""The main code to train a model"""

from datetime import datetime
from pathlib import Path

import random
import torch
import numpy as np
from torchinfo import summary

from multi_graph_cnn.utils import get_logger, load_config
from multi_graph_cnn.data import read_data, split_data, compute_the_laplacians
from multi_graph_cnn.model import MGCNN
from multi_graph_cnn.training import train_loop
from multi_graph_cnn.loss import rmse, DirichletReguLoss
from multi_graph_cnn.utils import sparse_mx_to_torch
from multi_graph_cnn.test import run_tests
from multi_graph_cnn.graph_insights import compute_laplacian_factor_from_model

from multi_graph_cnn.utils import get_tensorboard_writer



if __name__ == "__main__":

    config = load_config()
    log = get_logger("main", config.log_level)
    log.debug(config)

    seed = config.seed 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    data = read_data(config.dataset_name)
    O_training, O_target, O_test = split_data(data, config)
    L_row, L_col = compute_the_laplacians(data)

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = Path("saved_models/" + now)
    results_dir_path = dir_path / "results"
    config.output_dir = str(dir_path)
    config.result_dir = str(results_dir_path)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L_row = sparse_mx_to_torch(L_row).to(config.device)
    L_col = sparse_mx_to_torch(L_col).to(config.device)

    L_row_rescaled = L_row - torch.eye(L_row.shape[0], device=config.device)
    L_col_rescaled = L_col - torch.eye(L_col.shape[0], device=config.device)

    model = MGCNN(L_row_rescaled, L_col_rescaled, config)
    model = model.to(config.device)
    summary(
        model,
        depth=5,
        input_size=(data['M'].shape[0], data['M'].shape[1]),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        # weight_decay=config.weight_decay,
        )

    loss = DirichletReguLoss(L_row, L_col, config)
    loss_rmse = rmse

    data = torch.tensor(data["M"]).to(config.device)
    O_training = torch.tensor(O_training).to(config.device)
    O_test = torch.tensor(O_test).to(config.device)
    O_target = torch.tensor(O_target).to(config.device)

    # Initialize TensorBoard Writer
    writer = get_tensorboard_writer(config)
    log.info(f" TensorBoard logging enabled in: {config.output_dir}/{config.tensorboard_dir}")

    log.info("Starting training...")
    try:
        train_loop(model, data, O_training, O_target, O_test, optimizer, loss, loss_rmse, config, writer)
    except KeyboardInterrupt:
        log.warning("Training interrupted by user.")

    # Close the writer
    writer.close()

    # [OPTIONAL] Load the best model before running final tests
    best_model_path = f"{config.output_dir}/best_model.pth"
    try:
        model.load_state_dict(torch.load(best_model_path))
        log.info("🏆 Loaded best model for final testing.")
    except Exception as e:
        log.warning("Could not load best model (maybe none was saved?), using last state.")

    run_tests(model, data, O_training, O_target, O_test, loss, loss_rmse, config)

    compute_laplacian_factor_from_model(model, config)
    
    log.info("✅ Pipeline completed successfully")
