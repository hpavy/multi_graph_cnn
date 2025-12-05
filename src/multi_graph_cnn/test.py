"""In order to test the model"""


def find_loss_test(model, data, O_val, loss, loss_rmse, config):
    data_val = data * O_val
    predict = model(data_val)
    return loss(predict, data_val), loss_rmse(predict, data_val)

