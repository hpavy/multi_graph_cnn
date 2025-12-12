"""Create the model from the paper"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        nn.init.zeros_(self.b_f)
        nn.init.zeros_(self.b_i)
        nn.init.zeros_(self.b_o)
        nn.init.zeros_(self.b_c)
        nn.init.zeros_(self.b_out)

    def reset_hidden_states(self, batch_size=None): # Add optional argument
        if batch_size is not None:
            self.h = torch.zeros(batch_size, self.n_conv_feat).to(self.device)
            self.c = torch.zeros(batch_size, self.n_conv_feat).to(self.device)
        else:
            self.h = None
            self.c = None

    def forward(self, x_conv):
        """qxmxn -> mxn"""

        if self.h is None:
            self.h = torch.zeros([x_conv.shape[0], self.n_conv_feat]).to(
                self.device
            )  # m.nxq
            self.c = torch.zeros([x_conv.shape[0], self.n_conv_feat]).to(
                self.device
            )  # m.nxq

        f = F.sigmoid(
            torch.matmul(x_conv, self.W_f) + torch.matmul(self.h, self.U_f) + self.b_f
        )  # m.nxq
        i = F.sigmoid(
            torch.matmul(x_conv, self.W_i) + torch.matmul(self.h, self.U_i) + self.b_i
        )  # m.nxq
        o = F.sigmoid(
            torch.matmul(x_conv, self.W_o) + torch.matmul(self.h, self.U_o) + self.b_o
        )  # m.nxq

        update_c = F.sigmoid(
            torch.matmul(x_conv, self.W_c) + torch.matmul(self.h, self.U_c) + self.b_c
        )  # m.nxq
        self.c = torch.multiply(f, self.c) + torch.multiply(i, update_c)
        self.h = torch.multiply(o, F.sigmoid(self.c))

        delta_x = F.tanh(torch.matmul(self.c, self.W_out) + self.b_out)
        
        return delta_x


class BilinearChebConv(nn.Module):
    def __init__(self, L_row, L_col, config):
        """
        Bilinear Chebyshev Graph Convolution.

        Args:
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
        self.theta = nn.Parameter(
            torch.Tensor(self.p_row + 1, self.p_col + 1, self.out_channels)
        )

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
            basis[k] = 2 * torch.matmul(L, basis[k - 1]) - basis[k - 2]

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
        x_row_transformed = torch.einsum("imp, pn -> imn", self.Tr, x)
        # Step B: Transform Columns ((Tr * X) * Tc)
        # Note: We multiply by Tc^T implicitly because X is (M, N) and Tc is (N, N)
        # (p_row, c, m, n) @ (p_col, n, n) -> (p_row, p_col, c, m, n)
        basis_features = torch.einsum("imn, jnk -> ij mk", x_row_transformed, self.Tc)
        # Step C: Linearly combine using Theta
        # (p_row, p_col, c, m, n) * (p_row, p_col, c, o) -> (o, m, n)
        out = torch.einsum("ij  m n, ij  o -> o m n", basis_features, self.theta)
        # 3. Add Bias
        out = out + self.bias.view(-1, 1, 1)
        x_conv = F.relu(out)
        x_conv = x_conv.permute(1, 2, 0)  # mxnxq
        x_conv = x_conv.reshape(-1, x_conv.shape[-1])
        return x_conv


class MonoChebConv(nn.Module):
    def __init__(self, L, config):
        """
        Standard Chebyshev Graph Convolution (Mono-Graph).
        
        Args:
            in_channels (int): Dimension of input features (e.g., rank 'r' of factor W).
            out_channels (int): Dimension of output features (q).
            L (Tensor): The scaled Laplacian matrix (N x N).
            order (int): Chebyshev polynomial order (p).
            config: Configuration object containing device info.
        """
        super().__init__()
        self.device = config.device
        self.in_channels = config.rank
        self.out_channels = config.out_channels
        self.order = config.p_order

        # Theta: Learnable parameters for each polynomial order k
        # Shape: (Order+1, In_Channels, Out_Channels)
        self.theta = nn.Parameter(
            torch.Tensor(self.order + 1, self.in_channels, self.out_channels)
        )

        self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        
        # Pre-compute Chebyshev Basis Matrices (Static Graph Assumption)
        # Shape: (Order+1, N, N)
        self.T = self.compute_cheb_polynomials(L, self.order)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def compute_cheb_polynomials(self, L, order):
        """
        Computes a stack of Chebyshev polynomials T_k(L).
        Returns tensor of shape (order+1, N, N)
        """
        n = L.size(0)
        basis = torch.empty((order + 1, n, n), device=self.device)

        # T_0(L) = I
        basis[0] = torch.eye(n, device=self.device)

        if order > 0:
            # T_1(L) = L
            basis[1] = L

        for k in range(2, order + 1):
            # Recurrence: T_k = 2 * L * T_{k-1} - T_{k-2}
            basis[k] = 2 * torch.matmul(L, basis[k - 1]) - basis[k - 2]

        return basis

    def forward(self, x):
        """
        Args:
            x: Input feature matrix (N, In_Channels) 
               (e.g., Factor W of shape M x R)

        Returns:
            Output (N, Out_Channels)
        """
        # Einsum Explanation:
        # k: Chebyshev order (0..p)
        # n: Neighbor node index
        # m: Target node index (Output row)
        # i: Input feature channel
        # o: Output feature channel

        # 1. Graph Diffusion Step: Apply Basis T_k to Input X
        # (Order, N, N) @ (N, In) -> (Order, N, In)
        # T[k] @ x corresponds to diffusing features k-hops away
        support = torch.einsum("kmn, ni -> kmi", self.T, x)
        
        # 2. Feature Transformation Step: Apply Weights Theta
        # (Order, N, In) @ (Order, In, Out) -> (N, Out)
        # We sum over 'k' (combining different hop info) and 'i' (combining input features)
        out = torch.einsum("kmi, kio -> mo", support, self.theta)

        # 3. Bias & Nonlinearity
        out = out + self.bias
        out = F.relu(out)
        
        return out

class MGCNN(nn.Module):
    def __init__(self, L_row, L_col, config):
        super().__init__()
        self.device = config.device
        self.conv = BilinearChebConv(L_row, L_col, config)
        self.rnn = RNN(config)
        self.nb_iterations_rnn = config.nb_iterations_rnn

    def forward(self, x):
        """The shape of x is mxn"""
        self.rnn.reset_hidden_states()
        m,n = x.shape
        for _ in range(self.nb_iterations_rnn):
            x_conv = self.conv(x)  # qxmxn
            dx = self.rnn(x_conv)  # mxn
            dx = dx.flatten().reshape(m, n)
            x = x + dx  # mxn
        return x  # mxn

    def forward_all_diffusion_steps(self, x) -> list[torch.Tensor]:
        """The shape of x is mxn"""
        self.rnn.reset_hidden_states()
        m,n = x.shape
        
        list_diffusion_step = []
        for _ in range(self.nb_iterations_rnn):
            x_conv = self.conv(x)  # qxmxn
            dx = self.rnn(x_conv)  # mxn
            dx = dx.flatten().reshape(m, n)
            x = x + dx  # mxn
            
            list_diffusion_step.append(x)
        return list_diffusion_step  # self.nb_iterations_rnn x m x n


class sRGCNN(nn.Module):
    def __init__(self, L_row, L_col,  config):
        """
        Separable Recurrent Graph CNN (sRGCNN) - Algorithm 2 from the paper.
        
        Args:
            L_row: Laplacian for Users
            L_col: Laplacian for Items
            in_channels: The rank 'r' of the factorization (W has shape M x r)
            config: Configuration object
        """
        super().__init__()
        self.device = config.device
        self.nb_iterations_rnn = config.nb_iterations_rnn
        
        # 1. Convolutions
        # Row Conv (Users): Input 'r' -> Output 'q'
        self.conv_row = MonoChebConv(
            L=L_row,
            config=config
        )
        
        # Col Conv (Items): Input 'r' -> Output 'q'
        self.conv_col = MonoChebConv(
            L=L_col,
            config=config
        )

        # 2. Recurrent Networks
        # We need separate RNNs for W and H as they have different dynamics/sizes
        self.rnn_row = RNN(config)
        self.rnn_col = RNN(config)

    def forward(self, W, H):
        """
        Args:
            W: User Factors (M x r)
            H: Item Factors (N x r)
        Returns:
            W, H (Refined Factors)
        """
        M = W.shape[0]
        N = H.shape[0]
        
        # Reset RNN states for the new batch/diffusion process
        self.rnn_row.reset_hidden_states(M)
        self.rnn_col.reset_hidden_states(N)

        for _ in range(self.nb_iterations_rnn):
            # --- Row Update (Users) [cite: 236-241] ---
            # 1. Convolve W (Spatial)
            W_conv = self.conv_row(W)  # (M, q)
            # 2. RNN Step (Temporal)
            dW = self.rnn_row(W_conv)  # (M, r)
            # 3. Update
            W = W + dW

            # --- Col Update (Items) [cite: 225-230] ---
            # 1. Convolve H (Spatial)
            H_conv = self.conv_col(H)  # (N, q)
            # 2. RNN Step (Temporal)
            dH = self.rnn_col(H_conv)  # (N, r)
            # 3. Update
            H = H + dH

        return W, H

    def forward_all_diffusion_steps(self, W, H) -> list[torch.Tensor]:
        """
        Returns the full reconstructed matrix X = WH^T at every step.
        Used for visualization or monitoring convergence.
        """
        M = W.shape[0]
        N = H.shape[0]
        
        self.rnn_row.reset_hidden_states(M)
        self.rnn_col.reset_hidden_states(N)
        
        list_diffusion_step = []

        # Initial State
        list_diffusion_step.append(W @ H.t())

        for _ in range(self.nb_iterations_rnn):
            # Update W
            W_conv = self.conv_row(W)
            dW = self.rnn_row(W_conv)
            W = W + dW

            # Update H
            H_conv = self.conv_col(H)
            dH = self.rnn_col(H_conv)
            H = H + dH
            
            # Reconstruct Full Matrix X = W * H.T
            X_rec = W @ H.t()
            list_diffusion_step.append(X_rec)

        return list_diffusion_step