from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import comb
import numpy as np
from scipy.sparse.csgraph import shortest_path

from multi_graph_cnn.model import MGCNN
from multi_graph_cnn.utils import get_logger
from multi_graph_cnn.visualization import plot_coef_influence, plot_k_hop_influence

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
    log.debug("Mean influence of convolution (i,j) = i-hops neighbourhood row (user) influence with j-hops (items) influence")
    log.debug(absolute_value_influence)

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



def compute_exact_2d_influence(theta_2d, R_basis, C_basis, W_row, W_col, config):
    """
    Computes the EXACT spatial influence heatmap for the 2D MGCNN.
    
    Args:
        theta_2d (np.ndarray): The learned coefficients matrix (shape p_row x p_col).
        L_row, L_col (sparse matrix): The Laplacians for row (item) and col (user) graphs.
        max_dist_row, max_dist_col (int): The max distances to analyze.
        
    Returns:
        np.ndarray: A 2D heatmap (shape max_dist_row+1 x max_dist_col+1)
                    where entry [k, l] is the mean absolute influence of
                    interaction between k-hop items and l-hop users.
    """

    # 2. Compute Geodesic Shells (Distance Matrices)
    dist_r = shortest_path(W_row, directed=False, unweighted=True)
    dist_c = shortest_path(W_col, directed=False, unweighted=True)
    
    # 3. Compute Influence Heatmap
    heatmap = np.zeros((config.p_order_row + 1, config.p_order_col + 1))
    
    # We loop over the target "Distance Grid"
    for k in range(config.p_order_row + 1):
        for l in range(config.p_order_col + 1):
            
            # --- Extract Features for Row Shell k ---
            # Find all item pairs (v, v') at distance k
            mask_r = (dist_r == k)
            if np.sum(mask_r) == 0: continue
            
            # Extract the polynomial values for these pairs
            # V_r shape: (Num_Pairs_k, p_row)
            # R_basis[:, mask_r] gives (p_row, Num_Pairs_k) -> Transpose
            V_r = R_basis[:, mask_r].T 
            
            # --- Extract Features for Col Shell l ---
            # Find all user pairs (u, u') at distance l
            mask_c = (dist_c == l)
            if np.sum(mask_c) == 0: continue
            
            # V_c shape: (Num_Pairs_l, p_col)
            V_c = C_basis[:, mask_c].T
            
            # --- Compute Interaction Energy ---
            # The filter weight for a specific user-pair (u,u') and item-pair (v,v')
            # is given by the bilinear form: val = V_r[v,v'] @ Theta @ V_c[u,u'].T
            #
            # We want Mean(Abs(val)). 
            # Since V_r and V_c can be large, we compute the matrix product Z
            # Z shape: (Num_Pairs_k, Num_Pairs_l)
            
            # Optimization: If too large, sample pairs. 
            # Here we assume analysis on manageable subgraphs.
            
            Intermediate = V_r @ theta_2d # Shape (Pairs_k, p_col)
            Z = Intermediate @ V_c.T      # Shape (Pairs_k, Pairs_l)
            
            mean_energy = np.mean(np.abs(Z))
            heatmap[k, l] = mean_energy
            
    return heatmap

def compute_energy_k_distant_from_model(model, dataset, config):
    path_results = Path(config.result_dir)
    path_results.mkdir(exist_ok=True)
    polynom_tcheby_row = model.conv.Tr.detach().numpy()
    polynom_tcheby_col = model.conv.Tc.detach().numpy()
    heatmaps = []
    log.info("Processing energy of k-distant neigbour ...")
    for q in range(config.out_channels):
        coefs = model.conv.theta[:,:,q].detach().numpy()
        heatmap = compute_exact_2d_influence(coefs, polynom_tcheby_row, polynom_tcheby_col, dataset["W_row"], dataset["W_col"], config)
        heatmap_percent = heatmap / heatmap.sum()
        heatmaps.append(heatmap_percent)
        plot_k_hop_influence(heatmap_percent, f"Filter {q}", show=False)
        plt.savefig(path_results / f"heatmap_energy_filter_{q}.png")
        plt.close()

    mean = np.array(heatmaps).mean(axis=0)
    plot_k_hop_influence(mean, f"Mean across all", show=False)
    plt.savefig(path_results / f"heatmap_energy_mean.png")
    plt.close()
