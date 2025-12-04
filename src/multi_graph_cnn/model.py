"""Create the model from the paper"""

from torch import nn


class RNN(nn.Module):
    def init(self, config):
        self.device = config.device

    def forward(self, x):
        pass


class Conv(nn.Module):
    def init(self, config):
        self.device = config.device

    def forward(self, x):
        pass


class MGCNN(nn.Module):
    def init(self, config):
        self.device = config.device
        self.conv = Conv(config)
        self.rnn = RNN(config)
        self.nb_iterations_rnn = config.nb_iterations_rnn

    def forward(self, x):
        """The shape of x is mxn"""
        for iteration in range(self.nb_iterations_rnn):
            x = self.conv(x)  # mxnxq
            dx = self.rnn(x)  # mxn
            x = x + dx  # mxn
        return x  # mxn
