"""Create the model from the paper"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_conv_feat = config.out_channels_GCCN

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

    
    def reset_hiden_states(self):
        self.h = None
        self.c = None


    def forward(self, x_conv):
        x_conv = x_conv.permute(2, 0, 1)
        if self.h is None:
            self.h =  nn.zeros([x_conv.shape[0]*x_conv.shape[1], self.n_conv_feat])
            self.c = nn.zeros([x_conv.shape[0]*x_conv.shape[1], self.n_conv_feat])

        f = F.sigmoid(nn.matmul(x_conv, self.W_f) + nn.matmul(self.h, self.U_f) + self.b_f)
        i = F.sigmoid(nn.matmul(x_conv, self.W_i) + nn.matmul(self.h, self.U_i) + self.b_i)
        o = F.sigmoid(nn.matmul(x_conv, self.W_o) + nn.matmul(self.h, self.U_o) + self.b_o)
        
        update_c = F.sigmoid(nn.matmul(x_conv, self.W_c) + nn.matmul(self.h, self.U_c) + self.b_c)
        self.c = f @ self.c + i @ update_c
        self.h = o @ F.sigmoid(self.c)

        delta_x = nn.tanh(nn.matmul(self.c, self.W_out) + self.b_out)
        return delta_x


class BilinearChebConv(nn.Module):
    def __init__(self, Lr, Lc, config):
        """
        Bilinear Chebyshev Graph Convolution.
        
        Args:
            in_channels (int): Input features (usually 1 for a raw matrix).
            out_channels (int): Output dimension (q).
            p_order_row (int): Chebyshev order for rows (users).
            p_order_col (int): Chebyshev order for columns (items).
        """
        super(BilinearChebConv, self).__init__()
        self.in_channels = 1
        self.out_channels = config.out_channels_GCCN
        self.p_row = config.p_order_row
        self.p_col = config.p_order_col
        
        # Theta: Learnable parameters for every combination of row/col polynomial orders.
        # Shape: (P_row+1, P_col+1, In_Channels, Out_Channels)
        self.theta = nn.Parameter(torch.Tensor(
            self.p_row + 1, 
            self.p_col + 1, 
            1, 
            self.out_channels
        ))
        
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        self.reset_parameters()
        # 1. Pre-compute Chebyshev Basis Matrices
        # Shape: (p_row+1, M, M)
        self.Tr = self.compute_cheb_polynomials(Lr, self.p_row)
        # Shape: (p_col+1, N, N)
        self.Tc = self.compute_cheb_polynomials(Lc, self.p_col)

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
            x: Input matrix ( In_Channels, M, N)
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
        x_row_transformed = torch.einsum('imp, cpn -> icmn', self.Tr, x)
        
        # Step B: Transform Columns ((Tr * X) * Tc)
        # Note: We multiply by Tc^T implicitly because X is (M, N) and Tc is (N, N)
        # (p_row, c, m, n) @ (p_col, n, n) -> (p_row, p_col, c, m, n)
        basis_features = torch.einsum('icmn, jnk -> ijc mk', x_row_transformed, self.Tc)
        
        # Step C: Linearly combine using Theta
        # (p_row, p_col, c, m, n) * (p_row, p_col, c, o) -> (o, m, n)
        out = torch.einsum('ij c m n, ij c o -> o m n', basis_features, self.theta)
        
        # 3. Add Bias
        out = out + self.bias.view(-1, 1, 1)
        
        return out


class MGCNN(nn.Module):
    def init(self, Lr,Lc, config):
        self.device = config.device
        self.conv = BilinearChebConv(Lr, Lc, config)
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
