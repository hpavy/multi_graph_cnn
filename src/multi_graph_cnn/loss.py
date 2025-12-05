import torch
from torch import nn

def normalize_x(x, min_val=1, max_val=5):
    """
    Scales x to the range [min_val, max_val] based on its current min/max.
    Matches the TF notebook logic: 1 + 4 * (x - min) / (max - min)
    """
    curr_min = x.min()
    curr_max = x.max()
    
    # Avoid division by zero
    denom = curr_max - curr_min
    if denom < 1e-8:
        return x
        
    # Scale to 0-1 then to min_val-max_val
    return min_val + (max_val - min_val) * (x - curr_min) / denom

class DirichletReguLoss(nn.Module):
    def __init__(self, L_row, L_col,config):
        super().__init__()
        self.L_row = L_row
        self.L_col = L_col
        self.gamma =config.gamma

    def forward(self, X, Y):
        """
        X : matrix learnt
        Y : known entries, values of zero are considered to be unknown
        """

        dirichlet_row = torch.trace(X.T @ self.L_row @ X)
        dirichlet_col = torch.trace(X @ self.L_col @ X.T)

        X=normalize_x(X)
        mask = Y > 0
        regularization_term = torch.norm(mask * (X - Y))

        return self.gamma/2*(dirichlet_row + dirichlet_col) + regularization_term


def rmse(learnt, target):
    """Check distance between learnt and target for non zero value of target"""
    learnt_norm=normalize_x(learnt)
    mask = target != 0
    masked_x = mask * learnt_norm
    mse_fun = nn.MSELoss()
    return torch.sqrt(mse_fun(masked_x, target))
