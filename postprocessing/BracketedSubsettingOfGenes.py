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
parser.add_argument('--random_iterations', type=int, default=1000, help='Number of random iterations for finding best genes')
parser.add_argument('--data_root', type=str, help='Root directory for preprocessed data.',default='../preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/')
# parser.add_argument('--cmap_file', type=str, help='Path to the CMAP CSV file.',default='../preprocessing/preprocessed_data/CellPairs/drug_landmarks.csv')
parser.add_argument('--cmap_file', type=str, help='Path to the CMAP CSV file.',default='../../TranslationalModels/OmicTranslationBenchmark/preprocessing/preprocessed_data/cmap_all_genes_q1_tas03.csv')
parser.add_argument('--gene_size', type=int, default=978, help='Number of genes to build a gene space')
parser.add_argument('--no_folds', type=int, default=5, help='Number of cross-validation folds.')

args = parser.parse_args()
random_iterations = args.random_iterations
data_root = args.data_root
gene_size = args.gene_size

# Read data
cmap_og = pd.read_csv(args.cmap_file, index_col=0)
r_all = np.corrcoef(cmap_og.values.T)
np.fill_diagonal(r_all, np.nan)
mean_r = np.nanmean(r_all,axis=0)
mean_r_s = pd.Series(mean_r, index=cmap_og.columns)
top_genes = mean_r_s.nlargest(gene_size).index.values
top_scores = mean_r_s.nlargest(gene_size).values
ex = np.exp(top_scores - np.max(top_scores))
weights = ex / np.sum(ex)
cmap = cmap_og.loc[:,top_genes]
genes = cmap.columns.values
num_genes = int(0.5*gene_size)
samples = cmap.index.values
best_r = -1
diff = 100
for j in range(random_iterations):
    genes_1 = np.random.choice(genes, size=num_genes, replace=False, p=weights)
    genes_2 = np.setdiff1d(genes,genes_1)
    ## To allow for some more randomness if the best_r has converged remove the wegihts
    if diff<5 and j>200:
        weights = None
    r = np.corrcoef(cmap[genes_1].values.T,cmap[genes_2].values.T)
    r = np.mean(r)
    if r > best_r:
        diff = 100 * abs(r - best_r)/best_r
        best_r = r
        best_genes_1 = genes_1
        best_genes_2 = genes_2
        print2log(f'New best average correlation ({r}) found in iteration {j}')
np.save(data_root+'best_genes_1.npy',best_genes_1)
np.save(data_root+'best_genes_2.npy',best_genes_2)

## Now from all genes remove these genes and find the second best
print2log('Now from all genes remove these genes and find the second best')
remaining_genes = np.setdiff1d(cmap_og.columns.values,best_genes_1)
remaining_genes = np.setdiff1d(remaining_genes,best_genes_2)
cmap_og = cmap_og.loc[:,remaining_genes]
r_all = np.corrcoef(cmap_og.values.T)
np.fill_diagonal(r_all, np.nan)
mean_r = np.nanmean(r_all,axis=0)
mean_r_s = pd.Series(mean_r, index=cmap_og.columns)
top_genes = mean_r_s.nlargest(gene_size).index.values
top_scores = mean_r_s.nlargest(gene_size).values
ex = np.exp(top_scores - np.max(top_scores))
weights = ex / np.sum(ex)
cmap = cmap_og.loc[:,top_genes]
genes = cmap.columns.values
num_genes = int(0.5*gene_size)
samples = cmap.index.values
best_r = -1
diff = 100
for j in range(random_iterations):
    genes_1 = np.random.choice(genes, size=num_genes, replace=False, p=weights)
    genes_2 = np.setdiff1d(genes,genes_1)
    ## To allow for some more randomness if the best_r has converged remove the wegihts
    if diff<5 and j>200:
        weights = None
    r = np.corrcoef(cmap[genes_1].values.T,cmap[genes_2].values.T)
    r = np.mean(r)
    if r > best_r:
        diff = 100 * abs(r - best_r)/best_r
        best_r = r
        best_genes_1 = genes_1
        best_genes_2 = genes_2
        print2log(f'New best average correlation ({r}) found in iteration {j}')
np.save(data_root+'second_best_genes_1.npy',best_genes_1)
np.save(data_root+'second_best_genes_2.npy',best_genes_2)

## And repeat again for third best
print2log('And repeat again for third best')
remaining_genes = np.setdiff1d(cmap_og.columns.values,best_genes_1)
remaining_genes = np.setdiff1d(remaining_genes,best_genes_2)
cmap_og = cmap_og.loc[:,remaining_genes]
r_all = np.corrcoef(cmap_og.values.T)
np.fill_diagonal(r_all, np.nan)
mean_r = np.nanmean(r_all,axis=0)
mean_r_s = pd.Series(mean_r, index=cmap_og.columns)
top_genes = mean_r_s.nlargest(gene_size).index.values
top_scores = mean_r_s.nlargest(gene_size).values
ex = np.exp(top_scores - np.max(top_scores))
weights = ex / np.sum(ex)
cmap = cmap_og.loc[:,top_genes]
genes = cmap.columns.values
num_genes = int(0.5*gene_size)
samples = cmap.index.values
best_r = -1
diff = 100
for j in range(random_iterations):
    genes_1 = np.random.choice(genes, size=num_genes, replace=False, p=weights)
    genes_2 = np.setdiff1d(genes,genes_1)
    ## To allow for some more randomness if the best_r has converged remove the wegihts
    if diff<5 and j>200:
        weights = None
    r = np.corrcoef(cmap[genes_1].values.T,cmap[genes_2].values.T)
    r = np.mean(r)
    if r > best_r:
        diff = 100 * abs(r - best_r)/best_r
        best_r = r
        best_genes_1 = genes_1
        best_genes_2 = genes_2
        print2log(f'New best average correlation ({r}) found in iteration {j}')
np.save(data_root+'third_best_genes_1.npy',best_genes_1)
np.save(data_root+'third_best_genes_2.npy',best_genes_2)

### And repeat for fourth best
print2log('And repeat for fourth best')
remaining_genes = np.setdiff1d(cmap_og.columns.values,best_genes_1)
remaining_genes = np.setdiff1d(remaining_genes,best_genes_2)
cmap_og = cmap_og.loc[:,remaining_genes]
r_all = np.corrcoef(cmap_og.values.T)
np.fill_diagonal(r_all, np.nan)
mean_r = np.nanmean(r_all,axis=0)
mean_r_s = pd.Series(mean_r, index=cmap_og.columns)
top_genes = mean_r_s.nlargest(gene_size).index.values
top_scores = mean_r_s.nlargest(gene_size).values
ex = np.exp(top_scores - np.max(top_scores))
weights = ex / np.sum(ex)
cmap = cmap_og.loc[:,top_genes]
genes = cmap.columns.values
num_genes = int(0.5*gene_size)
samples = cmap.index.values
best_r = -1
diff = 100
for j in range(random_iterations):
    genes_1 = np.random.choice(genes, size=num_genes, replace=False, p=weights)
    genes_2 = np.setdiff1d(genes,genes_1)
    ## To allow for some more randomness if the best_r has converged remove the wegihts
    if diff<5 and j>200:
        weights = None
    r = np.corrcoef(cmap[genes_1].values.T,cmap[genes_2].values.T)
    r = np.mean(r)
    if r > best_r:
        diff = 100 * abs(r - best_r)/best_r
        best_r = r
        best_genes_1 = genes_1
        best_genes_2 = genes_2
        print2log(f'New best average correlation ({r}) found in iteration {j}')
np.save(data_root+'fourth_best_genes_1.npy',best_genes_1)
np.save(data_root+'fourth_best_genes_2.npy',best_genes_2)

print2log('Done')