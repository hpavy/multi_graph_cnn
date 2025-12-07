"""The visualization functions"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

def plot_coef_influence(coefs, title, ax=None, show=True):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax = sns.heatmap(coefs, annot=True, fmt=".1e", ax=ax)
    ax.xaxis.tick_top()
    ax.set_ylabel("i-hops on users graph")
    ax.set_xlabel("j-hops on items graph")
    ax.set_title(f"Influence of convolution coefficients filter ({title})")

    if show: plt.show()


def plot_matrix_completion(matrix, ax=None, show=True):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax = sns.heatmap(matrix, ax=ax)
    ax.xaxis.tick_top()
    ax.set_ylabel("Users")
    ax.set_xlabel("Items")

    if show: plt.show()

def plot_error_distribution(df_error:pd.DataFrame, groupby, ax=None, show=True):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    if groupby:
        for elt in df_error["split"].unique():
            ax.hist(df_error["split" == elt]["error"], density=True, alpha=0.3, label=elt)
    else:
        ax.hist(df_error["error"], density=True)


    if show: plt.show()

    
