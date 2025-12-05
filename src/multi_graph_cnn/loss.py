import torch
from torch import nn


class DirichletReguLoss(nn.Module):
    def __init__(self, Lrow, Lcol):
        super().__init__()
        self.Lrow = Lrow
        self.Lcol = Lcol

    def forward(self, X, Y):
        """
        X : matrix learnt
        Y : known entries, values of zero are considered to be unknown
        """

        dirichlet_row = torch.trace(X.T @ self.Lrow @ X)
        dirichlet_col = torch.trace(X @ self.Lcol @ X.T)

        mask = Y > 0
        regularization_term = torch.norm(mask * (X - Y))

        return dirichlet_row + dirichlet_col + regularization_term


def rmse(learnt, target):
    """Check distance between learnt and target for non zero value of target"""
    mask = target != 0
    masked_x = mask * learnt
    mse_fun = nn.MSELoss()
    return torch.sqrt(mse_fun(masked_x, target))
