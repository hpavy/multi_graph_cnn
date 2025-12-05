"""The training loop"""

from multi_graph_cnn.test import compute_val_loss
from multi_graph_cnn.utils import get_logger

log = get_logger()


def train_loop(model, data, O_training, O_val, O_test, optimizer, loss, loss_rmse, config):
    data = data.to(config.device)
    model.eval()
    loss_val, loss_val_rmse = compute_val_loss(
        model, data, O_training,  O_val, loss, loss_rmse, config
        )
    log.info(f"Step 0: val: {loss_val:.1e} - val predict: {loss_val_rmse:.1e}")
    data_training_init = data * O_training
    for i in range(1, config.n_epoch + 1):
        model.train()
        data_training = data_training_init.clone()
        loss_dir_row, loss_dir_col, loss_reg = loss(model(data_training), data * (O_training + O_val))
        loss_train = float(config.gamma)/2*(loss_dir_row + loss_dir_col) + loss_reg
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % config.log_each == 0:
            loss_test, loss_test_rmse = compute_val_loss(
                model, data, O_training, O_test, loss, loss_rmse, config
                )
            log.info(
                f"Step {i}: train: {loss_train.item():.1e} - test: {loss_test:.1e}"
                f" - test predict: {loss_test_rmse:.1e} - dirich row: {loss_dir_row:.1e}"
                f" - dirich col: {loss_dir_col:.1e} - reg_froeb {loss_reg:.1e}"
                )
