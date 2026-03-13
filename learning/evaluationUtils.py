from __future__ import division, print_function
import torch
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt

#Define custom metrics for evaluation
def r_square(y_true, y_pred):
    SS_res =  torch.sum((y_true - y_pred)**2)
    SS_tot = torch.sum((y_true - torch.mean(y_true))**2)
    return (1 - SS_res/(SS_tot + 1e-7))

def get_cindex(y_true, y_pred):
    g = torch.sub(y_pred.unsqueeze(-1), y_pred)
    g = (g == 0.0).type(torch.FloatTensor) * 0.5 + (g > 0.0).type(torch.FloatTensor)

    f = torch.sub(y_true.unsqueeze(-1), y_true) > 0.0
    f = torch.tril(f.type(torch.FloatTensor))

    g = torch.sum(g*f)
    f = torch.sum(f)

    return torch.where(g==0.0, torch.tensor(0.0), g/f)

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=0)
    my = torch.mean(y, dim=0)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym,dim=0)
    x_square_sum = torch.sum(xm * xm,dim=0)
    y_square_sum = torch.sum(ym * ym,dim=0)
    r_den = torch.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return r #torch.mean(r)


def pseudoAccuracy(y_true, y_pred, eps=1e-4):
    from sklearn.metrics import accuracy_score

    y_true[torch.where(torch.abs(y_true) < eps)] = 0
    y_true[torch.where(y_true < 0)] = -1
    y_true[torch.where(y_true > 0)] = 1

    y_pred[torch.where(torch.abs(y_pred) < eps)] = 0
    y_pred[torch.where(y_pred < 0)] = -1
    y_pred[torch.where(y_pred > 0)] = 1

    acc = []
    for i in range(y_true.shape[0]):
        acc.append(accuracy_score(y_true[i, :].numpy(), y_pred[i, :].numpy()))

    return acc

def pseudoPresicion(y_true, y_pred, eps=1e-4):
    from sklearn.metrics import precision_score

    y_true[torch.where(torch.abs(y_true) < eps)] = 0
    y_true[torch.where(y_true < 0)] = -1
    y_true[torch.where(y_true > 0)] = 1

    y_pred[torch.where(torch.abs(y_pred) < eps)] = 0
    y_pred[torch.where(y_pred < 0)] = -1
    y_pred[torch.where(y_pred > 0)] = 1
    
    prec = []
    for i in range(y_true.shape[0]):
        prec.append(precision_score(y_true[i, :].numpy(), y_pred[i, :].numpy(),average=None))

    return prec

def compare_target_vs_reference(df,target_name, reference_name, feature_col='TF',
                                features2show = ['P10914','Q00978','P42226','P42229','P51692','P40763','P52630','P42224','Q14765','P84022','P01106','P05412',
                                                 'O00716','Q14209','Q01094','Q16254','P04637'], 
                                model_col='set', r_col='r', folds_col='fold',alpha = 0.4,change_size=True,change_alpha=True):
    """
    Compares Pearson correlations across folds for shuffled and actual models, adjusts p-values,
    and plots a scatterplot.
    Parameters:
        df (pd.DataFrame): DataFrame containing columns for gene, set (e.g. shuffled/train/test), r values, and folds.
        features2show (np.ndarray): 1D array of feature names to highlight on the scatterplot.
        feature_col (str): Column name for the feature.
        model_col (str): Column name for the model set.
        r_col (str): Column name for Pearson correlation values.
        folds_col (str): Column name for folds.
        target_name (str): Name of the target set (e.g. shuffled).
        reference_name (str): Name of the reference set (e.g. test).
    Returns:
        results_df (pd.DataFrame): DataFrame with gene, average r, p-value, and adjusted p-value.
        fig (plt.Figure): Scatterplot figure with avg r vs -log10(p-adjusted).
    """
    
    # Split the dataframe into shuffled and actual data
    df_shuffled = df[df[model_col] == target_name]#.dropna(subset=['r'])
    df_shuffled['r'] = df_shuffled['r'].fillna(0)
    df_actual = df[df[model_col] == reference_name]#.dropna(subset=['r'])
    df_actual['r'] = df_actual['r'].fillna(0)

    features = df[feature_col].unique()
    results = []

    # Iterate through genes and perform t-tests across folds
    for feat in features:
        r_shuffled = df_shuffled[df_shuffled[feature_col] == feat][r_col]
        r_actual = df_actual[df_actual[feature_col] == feat][r_col]
        
        # Perform t-test comparing shuffled and actual
        _, p_value = mannwhitneyu(r_actual, r_shuffled,alternative='greater')
        
        # Compute average r across folds for actual model
        avg_r_actual = np.nanmean(r_actual)

        results.append({'feature': feat, 'avg_r': avg_r_actual, 'p_value': p_value})

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Adjust p-values using Benjamini-Hochberg correction
    results_df['p_adjusted'] = multipletests(results_df['p_value'], method='fdr_bh')[1]

    # Annotation for the plot
    significant = 100* (results_df['p_adjusted']<0.05).sum()/len(results_df)
    sig_label = f"total features\nbetter than shuffled:\n{significant:.2f}%"
    significant_2show = 100 * (results_df[np.isin(results_df.feature,features2show)]['p_adjusted']<0.05).sum()/len(features2show)
    important_label = f"important genes\nbetter than shuffled:\n{significant_2show:.2f}%"

    # Get the first two colors from the default Matplotlib color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_default = default_colors[0]  # First default color
    color_highlight = default_colors[1]  # Second default color
    # Set colors based on whether the gene will be highlighted cause it is important (in genes2show)
    colors = [color_highlight if feat in features2show else color_default for feat in results_df['feature']]
    if change_size:
        sizes = [30 if feat in features2show else 3 for feat in results_df['feature']]
    else:
        sizes = 25
    if change_alpha:
        alphas = [0.85 if feat in features2show else alpha for feat in results_df['feature']]
    else:
        alphas = alpha
    # Plot avg_r vs -log10(p_adjusted)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(results_df['avg_r'], -np.log10(results_df['p_adjusted']), color=colors, alpha=alphas,s=sizes)
    ax.set_xlabel('Average r across folds', fontsize=16)
    ax.set_ylabel('-log10(p-adjusted)', fontsize=16)
    ax.set_title('Per feature performance', fontsize=16)
    ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1.5)
    ax.text(ax.get_xlim()[0] + 0.03, np.floor(ax.get_ylim()[1]) - 0.4,sig_label, fontsize=12)
    ax.text(ax.get_xlim()[0] + 0.03, np.floor(ax.get_ylim()[1]) - 1.1,important_label, fontsize=12)
    plt.tight_layout()

    return results_df, fig