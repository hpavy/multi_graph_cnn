"""The training loop"""
import os
from multi_graph_cnn.test import compute_target_loss
from multi_graph_cnn.utils import get_logger
import torch 

log = get_logger()


def train_loop(model, data, O_training, O_target, O_test, optimizer, loss, loss_rmse, config, writer =None):
    data = data.to(config.device)
    model.eval()
    loss_target, loss_target_rmse = compute_target_loss(
        model, data, O_training,  O_target, loss, loss_rmse, config
        )
    log.info(f"Step 0: target: {loss_target:.1e} - target predict: {loss_target_rmse:.1e}")

    # Initialize best score for checkpointing
    best_test_rmse = float('inf')

    data_training_init = data * O_training
    for i in range(1, config.n_epoch + 1):
        model.train()
        data_training = data_training_init.clone()
        loss_dir_row, loss_dir_col, loss_reg = loss(model(data_training), data * (O_training + O_target))
        loss_train = float(config.gamma)/2*(loss_dir_row + loss_dir_col) + loss_reg
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()

        # --- TENSORBOARD: Log Training Loss ---
        if writer:
            writer.add_scalar('Loss/Train_Total', loss_train.item(), i)
            writer.add_scalar('Loss/Train_Reg_Frob', loss_reg.item(), i)
            writer.add_scalar('Loss/Train_Dirichlet', (loss_dir_row + loss_dir_col).item(), i)

        if i % config.log_each == 0:
            loss_test, loss_test_rmse = compute_target_loss(
                model, data, O_training, O_test, loss, loss_rmse, config
                )

            # --- TENSORBOARD: Log Test Metrics ---
            if writer:
                writer.add_scalar('Loss/Test_Total', loss_test.item(), i)
                writer.add_scalar('Metric/RMSE', loss_test_rmse.item(), i)

            log.info(
                f"Step {i}: train: {loss_train.item():.1e} - test: {loss_test:.1e}"
                f" - test predict: {loss_test_rmse:.1e} - dirich row: {loss_dir_row:.1e}"
                f" - dirich col: {loss_dir_col:.1e} - reg_froeb {loss_reg:.1e}"
                )

            # --- CHECKPOINTING: Save Best Model ---
            # We save only if the current test RMSE is lower than the best seen so far
            if loss_test_rmse < best_test_rmse:
                best_test_rmse = loss_test_rmse
                save_path = os.path.join(config.output_dir, "best_model.pth")
                torch.save(model.state_dict(), save_path)
                 log.info(f" New best model saved (RMSE: {best_test_rmse:.4f}) at step {i}")
