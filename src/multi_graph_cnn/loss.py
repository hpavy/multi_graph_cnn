import torch
from torch import nn


class DirichletReguLoss(nn.Module):
    def __init__(self, L_row, L_col):
        super().__init__()
        self.L_row = L_row
        self.L_col = L_col

    def forward(self, X, Y):
        """
        X : matrix learnt
        Y : known entries, values of zero are considered to be unknown
        """

        dirichlet_row = torch.trace(X.T @ self.L_row @ X)
        dirichlet_col = torch.trace(X @ self.L_col @ X.T)

        mask = Y > 0
        regularization_term = torch.norm(mask * (X - Y))

        return dirichlet_row + dirichlet_col + regularization_term


def rmse(learnt, target):
    """Check distance between learnt and target for non zero value of target"""
    mask = target != 0
    masked_x = mask * learnt
    mse_fun = nn.MSELoss()
    return torch.sqrt(mse_fun(masked_x, target))
