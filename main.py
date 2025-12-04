"""The main code to train a model"""

from datetime import datetime
from pathlib import Path

import torch
from torchinfo import summary

from multi_graph_cnn.utils import get_logger, load_config, compute_loss
from multi_graph_cnn.data import read_data, split_data, compute_the_laplacians
from multi_graph_cnn.model import MGCNN
from multi_graph_cnn.dataset import GraphDataset
from multi_graph_cnn.training import train_loop


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

    model = MGCNN(config)
    dataset_train = GraphDataset(data_train, config)
    dataset_test = GraphDataset(data_test, config)

    summary(
        model,
        depth=5,
        # input_size=(1, 1, N, config.timepoints_input), # TODO
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        )
    
    loss = compute_loss(config)

    log.info("Starting training...")
    try:
        train_loop(model, dataset_train, dataset_test, optimizer, loss, config)
    except KeyboardInterrupt:
        log.warning("Training interrupted by user.")

    log.info("✅ Pipeline completed successfully")
