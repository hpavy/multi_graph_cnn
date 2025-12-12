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
    def __init__(self, L_row, L_col, config):
        super().__init__()
        self.L_row = L_row
        self.L_col = L_col
        self.gamma = float(config.gamma)

    def forward(self, X, Y, split_components=True):
        """
        X : matrix learnt
        Y : known entries, values of zero are considered to be unknown
        """

        dirichlet_row = torch.trace(X.T @ self.L_row @ X)
        dirichlet_col = torch.trace(X @ self.L_col @ X.T)

        X = normalize_x(X)
        mask = Y > 0
        regularization_term = torch.norm(mask * (X - Y)) ** 2
        regularization_term = regularization_term / torch.sum(mask)
        if split_components:
            return dirichlet_row, dirichlet_col, regularization_term
        else:
            return self.gamma/2*(dirichlet_row + dirichlet_col) + regularization_term


def rmse(learnt, target):
    """Check distance between learnt and target for non zero value of target"""
    learnt_norm=normalize_x(learnt)
    mask = target != 0
    # Calculate squared error only on the mask
    diff = (learnt_norm - target) * mask
    mse = torch.sum(diff ** 2) / torch.sum(mask) # Divide by count of ratings, not matrix size
    
    return torch.sqrt(mse)

def compute_factorized_rmse(W, H, target_mask, target_data, loss_rmse):
    """Helper to compute RMSE for factorized model"""
    # Reconstruct full matrix
    prediction = W @ H.t()
    
    val = loss_rmse(prediction, target_data * target_mask)
    return val

class DirichletReguLossSRGCNN(nn.Module):
    def __init__(self, L_row, L_col, config):
        super().__init__()
        self.L_row = L_row
        self.L_col = L_col
        self.gamma = float(config.gamma)

    def forward(self, W,H, Y, split_components=True):
        """
        X : matrix learnt
        Y : known entries, values of zero are considered to be unknown
        """  
        # A. Geometric Smoothness (Dirichlet Energy on Factors)
        # trace(W^T * L_row * W)
        dirichlet_W = torch.trace(W.t() @ self.L_row @ W)
        # trace(H^T * L_col * H)
        dirichlet_H = torch.trace(H.t() @ self.L_col @ H)
        
        loss_geom = (dirichlet_W + dirichlet_H)

        # B. Reconstruction Loss (Frobenius Norm on masked entries)
        # X_rec = W * H^T
        X_rec = W @ H.t()
        
        # We normalize X_rec for stability in loss calculation (matching original loss.py logic)
        X_rec_norm = normalize_x(X_rec)
        
        # Target data is O_training + O_target (known entries)
        mask = Y > 0
        diff = mask * (X_rec_norm - Y)
        loss_reg = torch.norm(diff) ** 2 / torch.sum(mask)

        if split_components:
            return dirichlet_W, dirichlet_H, loss_reg
        else:
            return (self.gamma) / 2 * loss_geom + loss_reg
