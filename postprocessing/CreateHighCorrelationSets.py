import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

# Initialize argparse
parser = argparse.ArgumentParser(description="Run across all data and genes in cmap file.")
# Data and output paths
parser.add_argument('--random_iterations', type=int, default=5, help='Number of random times I select some the same genes to be included in both sets.')
parser.add_argument('--data_root', type=str, help='Root directory for preprocessed data.',default='../preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/')
parser.add_argument('--ratio_selected', type=float, default=0.3, help='Ratio of genes selected to be included in both sets.')
parser.add_argument('--cmap_file', type=str, help='Path to the CMAP CSV file.',default='../../TranslationalModels/OmicTranslationBenchmark/preprocessing/preprocessed_data/cmap_all_genes_q1_tas03.csv')
parser.add_argument('--no_folds', type=int, default=5, help='Number of cross-validation folds.')

args = parser.parse_args()
random_iterations = args.random_iterations
data_root = args.data_root
ratio_selected = args.ratio_selected

# Read data
best_genes_1 = np.load(data_root+'best_genes_1.npy',allow_pickle=True)
best_genes_2 = np.load(data_root+'best_genes_2.npy',allow_pickle=True)
all_genes = np.union1d(best_genes_1,best_genes_2)
cmap_og = pd.read_csv(args.cmap_file, index_col=0).loc[:,all_genes]
r_all = np.corrcoef(cmap_og.values.T)
r_all = pd.DataFrame(r_all)
r_all.index = cmap_og.columns
r_all.columns = cmap_og.columns
num_genes_1 = int( len(best_genes_1) * ratio_selected)
num_genes_2 = int( len(best_genes_2) * ratio_selected)
for j in range(random_iterations):
    best_genes_1_subsetted = np.random.choice(best_genes_1, size=(len(best_genes_1)-num_genes_2), replace=False)
    best_genes_2_subsetted = np.random.choice(best_genes_2, size=(len(best_genes_2)-num_genes_1), replace=False)
    g1 = np.random.choice(best_genes_1, size=num_genes_1, replace=False)
    g2 = np.random.choice(best_genes_2, size=num_genes_2, replace=False)
    genes_1 = np.union1d(best_genes_1_subsetted,g2)
    genes_2 = np.union1d(best_genes_2_subsetted,g1)
    # r = np.corrcoef(cmap_og[genes_1].values.T,cmap_og[genes_1].values.T)
    # r = np.mean(r)
    r = r_all.loc[genes_1,genes_2]
    mean_r = np.nanmean(r,axis=0).mean()
    print2log(f'Average correlation ({mean_r}) found in iteration {j}, with {len(genes_1)} genes in set 1 and {len(genes_2)} genes in set 2')
    np.save(data_root+f'high_correlation_set_1_iter{j}.npy',g1)
    np.save(data_root+f'high_correlation_set_2_iter{j}.npy',g2)

print2log('Done')