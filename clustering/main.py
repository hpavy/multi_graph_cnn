"""The main code to train a model"""

from datetime import datetime
from pathlib import Path

import argparse
import random
import torch
import numpy as np
import json
from torchinfo import summary

from multi_graph_cnn.utils import get_logger, load_config
from multi_graph_cnn.data import read_data, split_data, compute_the_laplacians
from multi_graph_cnn.model import sRGCNN
from multi_graph_cnn.training import  train_loop_sRGCNN
from multi_graph_cnn.loss import rmse, DirichletReguLossSRGCNN
from multi_graph_cnn.utils import sparse_mx_to_torch, get_svd_initialization
from multi_graph_cnn.test import run_tests

from multi_graph_cnn.utils import get_tensorboard_writer

from clustering import create_cluster, create_stochastic_graph, generate_data,clustering_accuracy, find_cluster_from_graph, cluster_Spectral_embedding
from utils import compute_the_laplacians, split_ranking


if __name__ == "__main__":
    config = load_config("clustering/clustering_config.yaml")
    log = get_logger("main", config.log_level)
    log.debug(config)

    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # creating the clusters and the graphs
    user_tastes, movie_kinds = create_cluster(config)
    graph_users = create_stochastic_graph(
        user_tastes, config.proba_within_users, config.nb_neighbour_users
        )
    graph_movies = create_stochastic_graph(
        movie_kinds, config.proba_within_movies, config.nb_neighbour_movies
        )
    accuracy ={}

    cluster_G_m = find_cluster_from_graph(graph_movies)
    cluster_G_u = find_cluster_from_graph(graph_users)
    accuracy["accuracy_u_init"] = clustering_accuracy(cluster_G_u, user_tastes)
    accuracy["accuracy_m_init"] = clustering_accuracy(cluster_G_m, movie_kinds)

    # creating the ranking
    ranking = generate_data(user_tastes, movie_kinds, config)

    O_training, O_target, O_test = split_ranking(ranking, config)
    L_row = compute_the_laplacians(graph_users)
    L_col = compute_the_laplacians(graph_movies)

    model_id = config.proba_within_users + config.proba_within_movies 
    dir_path = Path("clustering/saved_models/" + model_id)
    config.output_dir = str(dir_path)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Entraînement Recommender System (MGCNN vs sRGCNN)")

    model_type ="sRGCNN"

    L_row = sparse_mx_to_torch(L_row).to(config.device)
    L_col = sparse_mx_to_torch(L_col).to(config.device)

    L_row_rescaled = L_row - torch.eye(L_row.shape[0], device=config.device)
    L_col_rescaled = L_col - torch.eye(L_col.shape[0], device=config.device)

    W,H  = get_svd_initialization(ranking,config.rank, config.device)

    model = sRGCNN(L_row_rescaled, L_col_rescaled, config)
    model = model.to(config.device)
    summary(
        model,
        depth=5,
        input_data = [W,H],
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        # weight_decay=config.weight_decay,
        )

    loss = DirichletReguLossSRGCNN(L_row, L_col, config)
    loss_rmse = rmse

    data = torch.tensor(ranking).to(config.device)
    O_training = torch.tensor(O_training).to(config.device)
    O_test = torch.tensor(O_test).to(config.device)
    O_target = torch.tensor(O_target).to(config.device)

    if not dir_path.exist():
        # Initialize TensorBoard Writer
        writer = get_tensorboard_writer(config)
        log.info(f" TensorBoard logging enabled in: {config.output_dir}/{config.tensorboard_dir}")
        
        log.info("Starting training...")
        try:
            train_loop_sRGCNN(model, W,H,data, O_training, O_target, O_test, optimizer, loss, loss_rmse, config, writer)
        except KeyboardInterrupt:
            log.warning("Training interrupted by user.")

        # Close the writer
        writer.close()

    # [OPTIONAL] Load the best model before running final tests
    best_model_path = f"{config.output_dir}/best_model.pth"
    try:
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint)
        W = checkpoint['W']
        H = checkpoint['H']
        cluster_W = cluster_Spectral_embedding(W)
        cluster_H = cluster_Spectral_embedding(H)
        accuracy["accuracy_W"] = clustering_accuracy(cluster_W, user_tastes)
        accuracy["accuracy_H"] = clustering_accuracy(cluster_H, movie_kinds)
        with open(f"{config.output_dir}/accuracy.json", "w") as f:
            json.dump(accuracy, f, indent=4) 
        log.info("🏆 Loaded best model for final testing.")
        log.info(accuracy)
    except Exception as e:
        log.warning("Could not load best model (maybe none was saved?), using last state.")

    run_tests(model, data, O_training, O_target, O_test, loss, loss_rmse, config)
    
    log.info("✅ Pipeline completed successfully")
