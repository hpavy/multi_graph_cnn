"""In order to test the model"""


import torch

from multi_graph_cnn.utils import get_logger

log = get_logger()


def compute_val_loss(model, data, O_training, O_val, loss, loss_rmse, config):
    data_train = data * O_training
    predict = model(data_train)
    data_val = data * O_val
    predict_val = model(data_val)
    return loss(predict_val, data_val, split_components=False), loss_rmse(predict, data_val)


def run_tests(model, data, O_training, O_val, O_test, loss, loss_rmse, config):
    _, loss_test_just_train = compute_val_loss(
        model, data, O_training,
        torch.ones_like(O_training), loss, loss_rmse, config
        )
    log.info(f"Test data: predict on train: {loss_test_just_train:.2e}")

    _, loss_test_just_train_val = compute_val_loss(
        model, data, O_training + O_val,
        torch.ones_like(O_training), loss, loss_rmse, config
        )
    log.info(f"Test data: predict on train and val: {loss_test_just_train_val:.2e}")

    _, loss_test_rmse = compute_val_loss(
        model, data, O_training + O_val + O_test,
        torch.ones_like(O_training), loss, loss_rmse, config
        )
    log.info(f"Test data: predict on train and val and test: {loss_test_rmse:.2e}")
    
    diff_steps = model.forward_all_diffusion_steps(data * O_training)
    loss_diff_iter = []
    for diff_step in diff_steps: 
        loss_value = loss_rmse(data, diff_step)
        loss_diff_iter.append(f"{loss_value:.2e}")
    log.info(f"Test data: predict on train : RMSE after each diffusion step : {loss_diff_iter}")
    
