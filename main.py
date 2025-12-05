"""The main code to train a model"""

from datetime import datetime
from pathlib import Path

import torch
from torchinfo import summary

from multi_graph_cnn.utils import get_logger, load_config
from multi_graph_cnn.data import read_data, split_data, compute_the_laplacians
from multi_graph_cnn.model import MGCNN
from multi_graph_cnn.training import train_loop
from multi_graph_cnn.loss import rmse, DirichletReguLoss
from multi_graph_cnn.utils import sparse_mx_to_torch
from multi_graph_cnn.test import compute_val_loss



if __name__ == "__main__":

    config = load_config()
    log = get_logger("main", config.log_level)
    log.debug(config)

    data = read_data(config.dataset_name)
    O_training, O_val, O_test = split_data(data, config)
    L_row, L_col = compute_the_laplacians(data)

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = Path("saved_models/" + now)
    config.output_dir = str(dir_path)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L_row = sparse_mx_to_torch(L_row).to(config.device)
    L_col = sparse_mx_to_torch(L_col).to(config.device)

    L_row = L_row - torch.eye(L_row.shape[0], device=config.device)
    L_col = L_col - torch.eye(L_col.shape[0], device=config.device)

    model = MGCNN(L_row, L_col, config)
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

    loss = DirichletReguLoss(L_row, L_col,config)
    loss_rmse = rmse

    data = torch.tensor(data["M"]).to(config.device)
    O_training = torch.tensor(O_training).to(config.device)
    O_test = torch.tensor(O_test).to(config.device)
    O_val = torch.tensor(O_val).to(config.device)

    log.info("Starting training...")
    try:
        train_loop(model, data, O_training, O_val, O_test, optimizer, loss, loss_rmse, config)
    except KeyboardInterrupt:
        log.warning("Training interrupted by user.")

    loss_test, loss_test_rmse = compute_val_loss(
        model, data, O_training, O_test, loss, loss_rmse, config
        )
    log.info(f"Test data: test: {loss_test:.2e} - test predict: {loss_test_rmse:.2e}")

    log.info("✅ Pipeline completed successfully")
