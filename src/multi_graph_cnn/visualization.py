"""The visualization functions"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

def plot_coef_influence(coefs, title, ax=None, show=True):
    """Plot the influence of convolution coefficients on the graph convolution.
    Args:
        coefs (np.ndarray): Coefficients of the convolution.
        title (str): Title of the plot.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax = sns.heatmap(coefs, annot=True, fmt=".1e", ax=ax)
    ax.xaxis.tick_top()
    ax.set_ylabel("i-hops on users graph")
    ax.set_xlabel("j-hops on items graph")
    ax.set_title(f"Influence of convolution coefficients filter ({title})")

    if show: plt.show()


def plot_matrix_completion(matrix, title, ax=None, show=True):
    """Plot a the matrix completion as a heatmap
    Args:
        matrix (np.ndarray): The matrix to plot.
        title (str): Title of the plot.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax = sns.heatmap(matrix, ax=ax)
    ax.xaxis.tick_top()
    ax.set_ylabel("Users")
    ax.set_xlabel("Items")
    ax.set_title(f"Matrix completion - {title}")

    if show: plt.show()

def plot_matrix_completion_duo(matrix1, matrix2, title1, title2, ax=None, show=True):
    """Plot two matrices as a side-by-side heatmap.
    Args:
        matrix1 (np.ndarray): The first matrix to plot.
        matrix2 (np.ndarray): The second matrix to plot.
        title1 (str): Title of the first plot.
        title2 (str): Title of the second plot.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True."""
    if ax is None:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    else:
        (ax1, ax2) = ax

    ax1 = sns.heatmap(matrix1, ax=ax1)
    ax1.xaxis.tick_top()
    ax1.set_ylabel("Users")
    ax1.set_xlabel("Items")
    ax1.set_title(f"Matrix completion - {title1}")
    

    ax2 = sns.heatmap(matrix2, ax=ax2)
    ax2.xaxis.tick_top()
    ax2.set_ylabel("Users")
    ax2.set_xlabel("Items")
    ax2.set_title(f"Matrix completion - {title2}")

    if show: plt.show()

def plot_error_distribution(df_error:pd.DataFrame, groupby:bool=False, ax=None, show=True):
    """Plot the distribution of errors in the matrix completion.
    Args:
        df_error (pd.DataFrame): DataFrame with 2 columns "error" and "split"
        Note that split would probably be between "train", "target", "test" or "true"
        groupby (bool, optional): Whether to group by split. Defaults to False.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 9))

    if groupby:
        for split in df_error["split"].unique():
            error_split = df_error[df_error["split"] == split]["error"]
            label = f"{split} ({len(error_split)})"
            ax.hist(error_split, bins=50, density=True, alpha=0.3, label=label)
        plt.legend()
    else:
        ax.hist(df_error["error"], bins=50, density=True)
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of errors")

    if show: plt.show()

    
