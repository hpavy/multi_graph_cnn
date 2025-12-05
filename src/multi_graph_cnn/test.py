"""In order to test the model"""


def compute_val_loss(model, data, O_training, O_val, loss, loss_rmse, config):
    data_train = data * O_training
    predict = model(data_train)
    data_val = data * O_val
    predict_val = model(data_val)
    return loss(predict_val, data_val, split_components=False), loss_rmse(predict, data_val)
