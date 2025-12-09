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
from multi_graph_cnn.loss import rmse, DirichletReguLoss
from multi_graph_cnn.utils import sparse_mx_to_torch
from multi_graph_cnn.test import run_tests

from multi_graph_cnn.utils import get_tensorboard_writer

from clustering import create_cluster, create_stochastic_graph, generate_data
from utils import compute_the_laplacians, split_ranking
from create_clusters import load_arrays
from training import train_loop
from create_clusters import save_arrays

for P_WITHIN in [0.7, 0.6, 0.5, 1., 0.9, 0.4]:
    if __name__ == "__main__":

        config = load_config("find_impact_graph/impact_graph.yaml")
        config.proba_within_users = P_WITHIN
        log = get_logger("main", config.log_level)
        log.debug(config)

        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # creating the clusters and the graphs
        user_tastes, movie_kinds = load_arrays("find_impact_graph/data/data.npz")

        # creating the ranking
        ranking = generate_data(user_tastes, movie_kinds, config)
        O_training, O_target, O_test = split_ranking(ranking, config)

        graph_users = create_stochastic_graph(
            user_tastes, config.proba_within_users, config.nb_neighbour_users
            )
        graph_movies = create_stochastic_graph(
            movie_kinds, config.proba_within_movies, config.nb_neighbour_movies
            )

        L_row = compute_the_laplacians(graph_users)
        L_col = compute_the_laplacians(graph_movies)

        # now = datetime.now().strftime("%Y%m%d-%H%M%S")
        # dir_path = Path("clustering/saved_models/find_impact_graph/" + config.proba_within_users)
        config.output_dir = f"find_impact_graph/results/proba_{P_WITHIN}"
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        L_row = torch.tensor(L_row, device=config.device, dtype=torch.float32)
        L_col = torch.tensor(L_col, device=config.device, dtype=torch.float32)

        L_row_rescaled = L_row - torch.eye(L_row.shape[0], device=config.device)
        L_col_rescaled = L_col - torch.eye(L_col.shape[0], device=config.device)

        model = MGCNN(L_row_rescaled, L_col_rescaled, config)
        model = model.to(config.device)
        summary(
            model,
            depth=5,
            input_size=(ranking.shape[0], ranking.shape[1]),
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.learning_rate),
            )

        loss = DirichletReguLoss(L_row, L_col, config)
        loss_rmse = rmse

        data = torch.tensor(ranking, dtype=torch.float32).to(config.device)
        O_training = torch.tensor(O_training, dtype=torch.float32).to(config.device)
        O_test = torch.tensor(O_test, dtype=torch.float32).to(config.device)
        O_target = torch.tensor(O_target, dtype=torch.float32).to(config.device)

        # Initialize TensorBoard Writer
        writer = get_tensorboard_writer(config)
        log.info(f" TensorBoard logging enabled in: {config.output_dir}/{config.tensorboard_dir}")

        log.info("Starting training...")
        try:
            results = train_loop(model, data, O_training, O_target, O_test, optimizer, loss, loss_rmse, config, writer)
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
        np.savez(config.output_dir + "/loss.npz", loss_rmse=results)

        log.info("✅ Pipeline completed successfully")
