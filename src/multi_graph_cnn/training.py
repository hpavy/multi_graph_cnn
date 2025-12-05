"""The training loop"""

from multi_graph_cnn.test import find_loss_test
from multi_graph_cnn.utils import get_logger

log = get_logger()


def train_loop(model, data, O_training, O_val, O_test, optimizer, loss, loss_rmse, config):
    data = data.to(config.device)
    model.eval()
    loss_val, loss_val_rmse = find_loss_test(model, data, O_val, loss, loss_rmse, config)
    log.info(f"Step 0: val: {loss_val:.3f} - val predict: {loss_val_rmse:.3f}")
    for i in range(1, config.n_epoch + 1):
        log_loss = []
        model.train()
        data_training = data * O_training
        loss_train = loss(model(data_training))
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % config.log_each == 0:
            loss_val, loss_val_rmse = find_loss_test(model, data, O_val, loss, loss_rmse, config)
            log.info(
                f"Step {i}: train: {loss_train.item():.3f} - val: {loss_val:.3f} - val predict: {loss_val_rmse:.3f}"
                )
    loss_test, loss_test_rmse = find_loss_test(model, data, O_test, loss, loss_rmse, config)
    log.info(f"Test data: test: {loss_test:.3f} - test predict: {loss_test_rmse:.3f}")
