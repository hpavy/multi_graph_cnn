"""Create the model from the paper"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super().__init__()
        self.n_conv_feat = n_conv_feat

        # Forget gate
        self.W_f = nn.Parameter(torch.Tensor(n_conv_feat, n_conv_feat))
        self.U_f = nn.Parameter(torch.Tensor(n_conv_feat, n_conv_feat))
        self.b_f = nn.Parameter(torch.Tensor(n_conv_feat))

        # Input gate
        self.W_i = nn.Parameter(torch.Tensor(n_conv_feat, n_conv_feat))
        self.U_i = nn.Parameter(torch.Tensor(n_conv_feat, n_conv_feat))
        self.b_i = nn.Parameter(torch.Tensor(n_conv_feat))

        # Output gate
        self.W_o = nn.Parameter(torch.Tensor(n_conv_feat, n_conv_feat))
        self.U_o = nn.Parameter(torch.Tensor(n_conv_feat, n_conv_feat))
        self.b_o = nn.Parameter(torch.Tensor(n_conv_feat))

        # Cell gate
        self.W_c = nn.Parameter(torch.Tensor(n_conv_feat, n_conv_feat))
        self.U_c = nn.Parameter(torch.Tensor(n_conv_feat, n_conv_feat))
        self.b_c = nn.Parameter(torch.Tensor(n_conv_feat))

        # Output parameters
        self.W_out = nn.Parameter(torch.Tensor(n_conv_feat, 1)) 
        self.b_out = nn.Parameter(torch.Tensor(1, 1)) 

        # The hidden states
        self.h = None
        self.c = None

        # Initialize weights and biases
        self.reset_parameters()


    def reset_parameters(self):
        # Xavier/Glorot initialization for weights
        init.xavier_uniform_(self.W_f)
        init.xavier_uniform_(self.U_f)
        init.xavier_uniform_(self.W_i)
        init.xavier_uniform_(self.U_i)
        init.xavier_uniform_(self.W_o)
        init.xavier_uniform_(self.U_o)
        init.xavier_uniform_(self.W_c)
        init.xavier_uniform_(self.U_c)
        init.xavier_uniform_(self.W_out)

    
    def reset_hiden_states(self):
        self.h = None
        self.c = None


    def forward(self, features, hidden_states):
        """
        """
        num_elements, feature_size = features.shape
        if self.h is None:
            self.h = #TODO
            self.c = #TODO

        f = F.sigmoid(nn.matmul(x_conv, self.W_f) + nn.matmul(self.h, self.U_f) + self.b_f)
        i = F.sigmoid(nn.matmul(x_conv, self.W_i) + nn.matmul(self.h, self.U_i) + self.b_i)
        o = F.sigmoid(nn.matmul(x_conv, self.W_o) + nn.matmul(self.h, self.U_o) + self.b_o)
        
        update_c = F.sigmoid(nn.matmul(x_conv, self.W_c) + nn.matmul(self.h, self.U_c) + self.b_c)
        self.c = f @ self.c + i @ update_c
        self.h = o @ F.sigmoid(self.c)

        delta_x = nn.tanh(nn.matmul(self.c, self.W_out) + self.b_out)
        return delta_x


class Conv(nn.Module):
    def init(self, config):
        self.device = config.device

    def forward(self, x):
        """Shape of x: mxn"""
        pass


class MGCNN(nn.Module):
    def init(self, config):
        self.device = config.device
        self.conv = Conv(config)
        self.rnn = RNN(config)
        self.nb_iterations_rnn = config.nb_iterations_rnn

    def forward(self, x):
        """The shape of x is mxn"""
        self.rnn.reset()
        for _ in range(self.nb_iterations_rnn):
            x = self.conv(x)  # mxnxq
            dx = self.rnn(x)  # mxn
            x = x + dx  # mxn
        return x  # mxn
