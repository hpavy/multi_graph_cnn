"""Create the model from the paper"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_graph_cnn.utils import sparse_mx_to_torch


class RNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.n_conv_feat = config.out_channels

        # Forget gate
        self.W_f = nn.Parameter(torch.Tensor(self.n_conv_feat, self.n_conv_feat))
        self.U_f = nn.Parameter(torch.Tensor(self.n_conv_feat, self.n_conv_feat))
        self.b_f = nn.Parameter(torch.Tensor(self.n_conv_feat))

        # Input gate
        self.W_i = nn.Parameter(torch.Tensor(self.n_conv_feat, self.n_conv_feat))
        self.U_i = nn.Parameter(torch.Tensor(self.n_conv_feat, self.n_conv_feat))
        self.b_i = nn.Parameter(torch.Tensor(self.n_conv_feat))

        # Output gate
        self.W_o = nn.Parameter(torch.Tensor(self.n_conv_feat, self.n_conv_feat))
        self.U_o = nn.Parameter(torch.Tensor(self.n_conv_feat, self.n_conv_feat))
        self.b_o = nn.Parameter(torch.Tensor(self.n_conv_feat))

        # Cell gate
        self.W_c = nn.Parameter(torch.Tensor(self.n_conv_feat, self.n_conv_feat))
        self.U_c = nn.Parameter(torch.Tensor(self.n_conv_feat, self.n_conv_feat))
        self.b_c = nn.Parameter(torch.Tensor(self.n_conv_feat))

        # Output parameters
        self.W_out = nn.Parameter(torch.Tensor(self.n_conv_feat, 1))
        self.b_out = nn.Parameter(torch.Tensor(1, 1))

        # The hidden states
        self.h = None
        self.c = None

        # Initialize weights and biases
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier/Glorot initialization for weights
        nn.init.xavier_uniform_(self.W_f)
        nn.init.xavier_uniform_(self.U_f)
        nn.init.xavier_uniform_(self.W_i)
        nn.init.xavier_uniform_(self.U_i)
        nn.init.xavier_uniform_(self.W_o)
        nn.init.xavier_uniform_(self.U_o)
        nn.init.xavier_uniform_(self.W_c)
        nn.init.xavier_uniform_(self.U_c)
        nn.init.xavier_uniform_(self.W_out)

    def reset_hidden_states(self):
        self.h = None
        self.c = None

    def forward(self, x_conv):
        """qxmxn -> mxn"""
        q, m, n = x_conv.shape
        x_conv = x_conv.permute(1, 2, 0)  # mxnxq
        x_conv = x_conv.reshape(-1, x_conv.shape[-1])
        if self.h is None:
            self.h = torch.zeros([x_conv.shape[0], self.n_conv_feat]).to(self.device)  # m.nxq
            self.c = torch.zeros([x_conv.shape[0], self.n_conv_feat]).to(self.device)  # m.nxq

        f = F.sigmoid(torch.matmul(x_conv, self.W_f) + torch.matmul(self.h, self.U_f) + self.b_f)  # m.nxq
        i = F.sigmoid(torch.matmul(x_conv, self.W_i) + torch.matmul(self.h, self.U_i) + self.b_i)  # m.nxq
        o = F.sigmoid(torch.matmul(x_conv, self.W_o) + torch.matmul(self.h, self.U_o) + self.b_o)  # m.nxq

        update_c = F.sigmoid(torch.matmul(x_conv, self.W_c) + torch.matmul(self.h, self.U_c) + self.b_c)  # m.nxq
        self.c = torch.multiply(f, self.c) + torch.multiply(i, update_c)
        self.h = torch.multiply(o, F.sigmoid(self.c))

        delta_x = F.tanh(torch.matmul(self.c, self.W_out) + self.b_out)
        delta_x = delta_x.flatten().reshape(m, n)
        return delta_x


class BilinearChebConv(nn.Module):
    def __init__(self, L_row, L_col, config):
        """
        Bilinear Chebyshev Graph Convolution.

        Args:
            in_channels (int): Input features (usually 1 for a raw matrix).
            out_channels (int): Output dimension (q).
            p_order_row (int): Chebyshev order for rows (users).
            p_order_col (int): Chebyshev order for columns (items).
        """
        super().__init__()
        self.device = config.device
        self.out_channels = config.out_channels
        self.p_row = config.p_order_row
        self.p_col = config.p_order_col

        # Theta: Learnable parameters for every combination of row/col polynomial orders.
        # Shape: (P_row+1, P_col+1, In_Channels, Out_Channels)
        self.theta = nn.Parameter(torch.Tensor(
            self.p_row + 1,
            self.p_col + 1,
            self.out_channels
        ))

        self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        self.reset_parameters()
        # 1. Pre-compute Chebyshev Basis Matrices
        # Shape: (p_row+1, M, M)
        self.Tr = self.compute_cheb_polynomials(L_row, self.p_row)
        # Shape: (p_col+1, N, N)
        self.Tc = self.compute_cheb_polynomials(L_col, self.p_col)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def compute_cheb_polynomials(self, L, order):
        """
        Computes a stack of Chebyshev polynomials T_k(L).
        Returns tensor of shape (order+1, N, N)
        """
        n = L.size(0)
        # Tensor to store [T_0, T_1, ..., T_k]
        basis = torch.empty((order + 1, n, n), device=L.device)

        # T_0(L) = I
        basis[0] = torch.eye(n, device=L.device)

        if order > 0:
            # T_1(L) = L (Assuming L is already scaled)
            basis[1] = L

        for k in range(2, order + 1):
            # Recurrence: T_k = 2 * L * T_{k-1} - T_{k-2}
            basis[k] = 2 * torch.matmul(L, basis[k-1]) - basis[k-2]

        return basis

    def forward(self, x):
        """
        Args:
            x: Input matrix (  M, N)
            L_row: Scaled Row Laplacian (M, M)
            L_col: Scaled Col Laplacian (N, N)

        Returns:
            Output (Out_Channels, M, N)
        """


        # 2. Compute the Bilinear Convolution using Einsum
        # This performs: Sum( Theta * (Tr @ X @ Tc) )

        # Explanation of indices:
        # i: row polynomial order
        # j: col polynomial order
        # c: input channels
        # o: output channels (q)
        # m, p: row dimensions (m x m)
        # n, k: col dimensions (n x n)

        # Step A: Transform Rows (Tr * X)
        # (p_row, m, m) @ (c, m, n) -> (p_row, c, m, n)
        x_row_transformed = torch.einsum('imp, pn -> imn', self.Tr, x)
        # Step B: Transform Columns ((Tr * X) * Tc)
        # Note: We multiply by Tc^T implicitly because X is (M, N) and Tc is (N, N)
        # (p_row, c, m, n) @ (p_col, n, n) -> (p_row, p_col, c, m, n)
        basis_features = torch.einsum('imn, jnk -> ij mk', x_row_transformed, self.Tc)
        # Step C: Linearly combine using Theta
        # (p_row, p_col, c, m, n) * (p_row, p_col, c, o) -> (o, m, n)
        out = torch.einsum('ij  m n, ij  o -> o m n', basis_features, self.theta)
        # 3. Add Bias
        out = out + self.bias.view(-1, 1, 1)
        return out


class MGCNN(nn.Module):
    def __init__(self, L_row, L_col, config):
        super().__init__()
        self.device = config.device
        L_row = sparse_mx_to_torch(L_row).to(self.device)
        L_col = sparse_mx_to_torch(L_col).to(self.device)
        self.conv = BilinearChebConv(L_row, L_col, config)
        self.rnn = RNN(config)
        self.nb_iterations_rnn = config.nb_iterations_rnn

    def forward(self, x):
        """The shape of x is mxn"""
        self.rnn.reset_hidden_states()
        for _ in range(self.nb_iterations_rnn):
            x_conv = self.conv(x)  # qxmxn
            dx = self.rnn(x_conv)  # mxn
            x = x + dx  # mxn
        return x  # mxn
