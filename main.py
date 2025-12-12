"""The main code to train a model"""

from datetime import datetime
from pathlib import Path

import argparse
import random
import torch
import numpy as np
from torchinfo import summary

from multi_graph_cnn.result_insights import compare_prediction
from multi_graph_cnn.utils import get_logger, load_config
from multi_graph_cnn.data import read_data, split_data, compute_the_laplacians
from multi_graph_cnn.model import MGCNN, sRGCNN
from multi_graph_cnn.training import train_loop, train_loop_sRGCNN
from multi_graph_cnn.loss import rmse, DirichletReguLoss, DirichletReguLossSRGCNN
from multi_graph_cnn.utils import sparse_mx_to_torch,get_svd_initialization
from multi_graph_cnn.test import run_tests
from multi_graph_cnn.graph_insights import compute_energy_k_distant_from_model, compute_laplacian_factor_from_model

from multi_graph_cnn.utils import get_tensorboard_writer



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Trainingr Recommender System (MGCNN vs sRGCNN)")
    
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        choices=["MGCNN", "sRGCNN"], 
        default=None,
        help="Model type to use: 'MGCNN' ou 'sRGCNN' (factorized)."
    )
    
    parser.add_argument("--epochs", type=int, default=None, help="Change epochs number")
    parser.add_argument("--lr", type=float, default=None, help="Change learning rate")

    args = parser.parse_args()


    if args.model == "sRGCNN":

        model_type ="sRGCNN"
        config = load_config("config_sRGCNN.yaml")
        log = get_logger("main", config.log_level)
        log.info(f" Launch pipeline for model : {model_type}")
        log.debug(config)
    
        if args.epochs:
            config.n_epoch = args.epochs
            
        if args.lr:
            config.learning_rate = args.lr

        seed = config.seed 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        dataset = read_data(config.dataset_name)
        O_training, O_target, O_test = split_data(dataset, config)
        L_row, L_col = compute_the_laplacians(dataset)

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_path = Path("saved_models/" + model_type +"/"+now)
        results_dir_path = dir_path / "results"
        config.output_dir = str(dir_path)
        config.result_dir = str(results_dir_path)
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        L_row = sparse_mx_to_torch(L_row).to(config.device)
        L_col = sparse_mx_to_torch(L_col).to(config.device)

        L_row_rescaled = L_row - torch.eye(L_row.shape[0], device=config.device)
        L_col_rescaled = L_col - torch.eye(L_col.shape[0], device=config.device)

        W,H  = get_svd_initialization(dataset['M'],config.rank, config.device)

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

        data = torch.tensor(dataset["M"]).to(config.device)
        O_training = torch.tensor(O_training).to(config.device)
        O_test = torch.tensor(O_test).to(config.device)
        O_target = torch.tensor(O_target).to(config.device)

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
            model.load_state_dict(torch.load(best_model_path))
            log.info("🏆 Loaded best model for final testing.")
        except Exception as e:
            log.warning("Could not load best model (maybe none was saved?), using last state.")

        
        log.info("✅ Pipeline completed successfully")
        
    else:

        model_type ="MGCNN"
        config = load_config("config_MGCNN.yaml")
        log = get_logger("main", config.log_level)
        log.info(f"Launch pipeline for model : {model_type}")
        log.debug(config)
    
        if args.epochs:
            config.n_epoch = args.epochs
            
        if args.lr:
            config.learning_rate = args.lr

        seed = config.seed 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        dataset = read_data(config.dataset_name)
        O_training, O_target, O_test = split_data(dataset, config)
        L_row, L_col = compute_the_laplacians(dataset)



        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_path = Path("saved_models/" + model_type +"/"+now)
        results_dir_path = dir_path / "results"
        config.output_dir = str(dir_path)
        config.result_dir = str(results_dir_path)
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        L_row = sparse_mx_to_torch(L_row).to(config.device)
        L_col = sparse_mx_to_torch(L_col).to(config.device)

        if config.enable_eigenvalue_scaling_laplacian:
            lambda_n_row = sorted(np.linalg.eigvals(L_row))[-1]
            lambda_n_col = sorted(np.linalg.eigvals(L_col))[-1]
        else:
            lambda_n_row = 2.0
            lambda_n_col = 2.0
        L_row_rescaled =  (2 / lambda_n_row) *L_row - torch.eye(L_row.shape[0], device=config.device)
        L_col_rescaled = (2 / lambda_n_col) * L_col - torch.eye(L_col.shape[0], device=config.device)

        model = MGCNN(L_row_rescaled, L_col_rescaled, config)
        model = model.to(config.device)
        summary(
            model,
            depth=5,
            input_size=(dataset['M'].shape[0], dataset['M'].shape[1]),
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.learning_rate),
            # weight_decay=config.weight_decay,
            )

        loss = DirichletReguLoss(L_row, L_col, config)
        loss_rmse = rmse

        data = torch.tensor(dataset["M"]).to(config.device)
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

        compare_prediction(model, data, O_training,O_target,O_test,  config)

        compute_energy_k_distant_from_model(model, dataset, config)
    
        log.info("✅ Pipeline completed successfully")
