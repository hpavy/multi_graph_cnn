from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from multi_graph_cnn.utils import get_logger
from multi_graph_cnn.loss import normalize_x
from multi_graph_cnn.visualization import plot_matrix_completion, plot_matrix_completion_duo, plot_error_distribution

log = get_logger()

def compare_prediction(model, data, O_training, O_target, O_test, config):
    path_results = Path(config.result_dir)
    
    log.info("Plotting true values vs. prediction (and saving fig)")
    # savefig prediction compared to rela one 
    predict = model(data * O_training)
    predict = normalize_x(predict)
    predict = predict.detach().cpu().numpy()
    data = data.detach().cpu().numpy()
    O_training = O_training.detach().cpu().numpy()
    O_test = O_test.detach().cpu().numpy()
    O_target = O_target.detach().cpu().numpy()
    plot_matrix_completion_duo(data, predict, "True values", "Prediction", show=False)
    plt.savefig(path_results / "heatmap_comp_real_value.png")
    plt.close()
    # difference 
    diff:np.ndarray = data - predict
    mean_abs_value = abs(diff).mean()
    plot_matrix_completion(diff, "Diff (Mean absolute error={mean_abs_value:.2f})", show=False)
    plt.savefig(path_results / "heatmap_diff_real_value.png")
    plt.close()

    log.info("Plotting error distributions (and saving fig)")
    # plot error distribution 
    split_matrix = np.full_like(data, fill_value="true", dtype=np.dtypes.StringDType())
    split_matrix[O_training == 1] = "train"
    split_matrix[O_target == 1] = "target"
    split_matrix[O_test == 1] = "test"
    df_error = pd.DataFrame({
        "error" : diff.flatten(),
        "split" : split_matrix.flatten()
    })
    plot_error_distribution(df_error, show=False)
    plt.savefig(path_results / "distribution_error.png")
    plt.close()
    plot_error_distribution(df_error, groupby=True, show=False)
    plt.savefig(path_results / "distribution_error_grouped.png")
    plt.close()
    