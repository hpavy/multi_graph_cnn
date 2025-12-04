"""The training loop"""

from torch.utils.data import DataLoader

from multi_graph_cnn.test import find_loss_test
from multi_graph_cnn.utils import get_logger

log = get_logger()


def train_loop(model, dataset_train, dataset_test, optimizer, loss, config):
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)

    log.info(f"Step 0: val: {find_loss_test(model, test_loader, config)}")
    for i in range(1, config.n_epoch + 1):
        log_loss = []
        model.train()
        for x_train, y_train in train_loader:
            x_train = x_train.to(config.device)
            y_train = y_train.to(config.device)
            _, loss = model(x_train, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Log
            log_loss.append(loss.item())
        loss_test = find_loss_test(model, test_loader, config)

        if i % config.log_each == 0:
            log.info(
                f"Step {i}: train: {(sum(log_loss) / len(log_loss)):.3f} - test: {loss_test:.3f}"
                )
