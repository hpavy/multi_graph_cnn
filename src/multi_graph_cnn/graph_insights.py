from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import comb

from multi_graph_cnn.model import MGCNN
from multi_graph_cnn.utils import get_logger
from multi_graph_cnn.visualization import plot_coef_influence

log = get_logger()

def compute_canonical_coefficients(theta_cheby):
    """
    Converts 2D Chebyshev coefficients to Canonical (Power basis) coefficients.
    
    Args:
        theta_cheby (np.ndarray): A 2D matrix of shape (p+1, p+1) containing 
                                  the learned parameters theta_jj'.
                                  Rows correspond to row-graph orders (j),
                                  Cols correspond to col-graph orders (j').
                                  
    Returns:
        np.ndarray: A 2D matrix of shape (p+1, p+1) containing the coefficients
                    for the powers of the Laplacian (L^a * L^b).
    """
    p_rows, p_cols = theta_cheby.shape
    p = max(p_rows, p_cols)
    
    # 1. Pre-compute Chebyshev polynomial coefficients in the power basis
    # T_k(x) = sum( c[n] * x^n )
    # We store these as a list of arrays, where index n corresponds to x^n
    cheby_polys = []
    
    # T_0(x) = 1
    t0 = np.zeros(p)
    t0[0] = 1
    cheby_polys.append(t0)
    
    if p > 1:
        # T_1(x) = x
        t1 = np.zeros(p)
        t1[1] = 1
        cheby_polys.append(t1)
        
        # Recurrence: T_k(x) = 2x * T_{k-1}(x) - T_{k-2}(x)
        for k in range(2, p):
            t_prev = cheby_polys[k-1] # T_{k-1}
            t_prev2 = cheby_polys[k-2] # T_{k-2}
            
            # Multiply T_{k-1} by 2x (shift coefficients right by 1 and multiply by 2)
            term1 = np.roll(t_prev, 1) * 2
            term1[0] = 0 # Clear wrap-around artifact if any
            
            # Subtract T_{k-2}
            t_curr = term1 - t_prev2
            cheby_polys.append(t_curr)

    # 2. Distribute theta weights into the canonical grid
    theta_canonical = np.zeros_like(theta_cheby)
    
    # Loop over every entry in the learned theta matrix
    for j in range(p_rows):       # Row-graph Chebyshev order
        for j_prime in range(p_cols): # Col-graph Chebyshev order
            
            weight = theta_cheby[j, j_prime]
            
            # Get the power expansion for T_j (row) and T_j' (col)
            # coeffs_r[a] is the coeff for x^a in T_j
            coeffs_r = cheby_polys[j] 
            coeffs_c = cheby_polys[j_prime]
            
            # Outer product: (sum a x^a) * (sum b y^b) = sum a,b (coeff_a * coeff_b) x^a y^b
            contribution = weight * np.outer(coeffs_r[:p_rows], coeffs_c[:p_cols])
            
            # Accumulate into the final canonical matrix
            theta_canonical += contribution
            
    return theta_canonical


def convert_rescaled_to_true_laplacian_2d(coeffs_rescaled_2d, lambda_max_row, lambda_max_col):
    """
    Converts 2D coefficients from the Rescaled Laplacian basis (L_tilde_r^a * L_tilde_c^b)
    to the True Laplacian basis (L_r^k * L_c^l).
    
    Args:
        coeffs_rescaled_2d (np.ndarray): 2D array of shape (p+1, p+1) where entry [a, b] 
                                         is the weight for L_tilde_r^a * L_tilde_c^b.
                                         (Output of compute_canonical_coefficients)
        lambda_max_row (float): The largest eigenvalue of the row graph Laplacian.
        lambda_max_col (float): The largest eigenvalue of the column graph Laplacian.
        
    Returns:
        np.ndarray: 2D array of shape (p+1, p+1) containing weights for L_r^k * L_c^l.
    """
    rows, cols = coeffs_rescaled_2d.shape
    p_row = rows - 1
    p_col = cols - 1
    
    # Calculate scaling factors alpha = 2 / lambda_max
    # [cite_start]Based on paper definition: L_tilde = (2/lambda_n)*L - I  [cite: 139]
    alpha_r = 2.0 / lambda_max_row
    alpha_c = 2.0 / lambda_max_col
    
    coeffs_true = np.zeros_like(coeffs_rescaled_2d)
    
    # Iterate over target powers k (row) and l (col)
    # These represent the exponents in the True Laplacian basis: L_r^k * L_c^l
    for k in range(p_row + 1):
        for l in range(p_col + 1):
            
            sum_val = 0.0
            
            # Sum over all "source" powers a and b from the Rescaled basis
            # We must have a >= k and b >= l to contribute
            for a in range(k, p_row + 1):
                for b in range(l, p_col + 1):
                    
                    # Get original weight
                    c_ab = coeffs_rescaled_2d[a, b]
                    
                    # Row expansion part: binomial(a, k) * (-1)^(a-k)
                    term_row = comb(a, k) * ((-1)**(a - k))
                    
                    # Col expansion part: binomial(b, l) * (-1)^(b-l)
                    term_col = comb(b, l) * ((-1)**(b - l))
                    
                    sum_val += c_ab * term_row * term_col
            
            # Apply the scaling factors alpha_r^k and alpha_c^l
            coeffs_true[k, l] = sum_val * (alpha_r**k) * (alpha_c**l)
            
    return coeffs_true


def compute_laplacian_factor_from_model(model:MGCNN, config):
    """
    The goal is to extract the paramaters from the model to interpret 
    the convolution in the graph domain 
    """
    # Get convolution paramaters 
    coef_cheby = model.conv.theta.detach().cpu().numpy()
    result_coefs = []
    for q in range(config.out_channels):
        # Compute coefficients in the canonical basis 
        coef_canonical_rescaled = compute_canonical_coefficients(coef_cheby[:,:,q])
        # Convert coeffecients for rescaled laplacian to normalized one
        # For now lambda_max is set to 2 per default, if computed enter them here
        coef_conv_laplacian = convert_rescaled_to_true_laplacian_2d(coef_canonical_rescaled, 2, 2)
        result_coefs.append(coef_conv_laplacian)

    result_coefs_arr = np.array(result_coefs)
    absolute_value_influence = abs(result_coefs_arr).mean(axis=0)
    log.info("Mean influence of convolution (i,j) = i-hops neighbourhood row (user) influence with j-hops (items) influence")
    log.info(absolute_value_influence)

    log.info("Processing plots of coefficient influence...")
    path_results = Path(config.result_dir)
    path_results.mkdir(exist_ok=True)
    plot_coef_influence(absolute_value_influence, "Mean absolute", show=False)
    plt.savefig(path_results / "heatmap_coef_mean_absolute.png")
    plt.close()
    for q, coefs in enumerate(result_coefs):
        plot_coef_influence(coefs, f"Filter {q}", show=False)
        plt.savefig(path_results / f"heatmap_coef_filter_{q}.png")
        plt.close()

    return result_coefs

