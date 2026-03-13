import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

# Initialize argparse
parser = argparse.ArgumentParser(description="Run cross-fold validation for comparing with AutoTransOP.")
# Data and output paths
parser.add_argument('--cell_lines', metavar='N', type=str, nargs='*', help='cell lines to artificially create different inputs',
                    default=["PC3","HT29","MCF7","A549","NPC","HEPG2","A375","YAPC","U2OS","MCF10A","HA1E","HCC515","ASC","VCAP","HUVEC","HELA"])
parser.add_argument('--random_iterations', type=int, default=5, help='Number of random iterations for imputing different inputs.')
parser.add_argument('--data_root', type=str, help='Root directory for preprocessed data.',default='../preprocessing/preprocessed_data/SameCellimputationModel/')
parser.add_argument('--cmap_file', type=str, help='Path to the CMAP CSV file.',default='../preprocessing/preprocessed_data/CellPairs/drug_landmarks.csv')
parser.add_argument('--output_dir', type=str, help='Directory to save output results.',default='./')
parser.add_argument('--no_folds', type=int, default=5, help='Number of cross-validation folds.')

args = parser.parse_args()
cell_lines = args.cell_lines
output_dir = args.output_dir
random_iterations = args.random_iterations
data_root = args.data_root
Path(output_dir).mkdir(parents=True, exist_ok=True)


# Read data
cmap = pd.read_csv(args.cmap_file, index_col=0)
genes = cmap.columns.values
gene_size = len(cmap.columns)
samples = cmap.index.values
r_all = pd.DataFrame({})
for cell in cell_lines:
    print2log(f'Cell line to create artificially different inputs: {cell}')
    for j in range(random_iterations):
        genes_1 = np.load(data_root+'genes_1'+cell+'_iter'+str(j)+'.npy',allow_pickle=True)
        genes_2 = np.setdiff1d(cmap.columns.values,genes_1)
        for fold_id in range(args.no_folds):
            trainInfo = pd.read_csv(data_root+cell+'/train_'+str(fold_id)+'.csv',index_col=0)
            valInfo = pd.read_csv(data_root+cell+'/val_'+str(fold_id)+'.csv',index_col=0)
            cmap_train = cmap.loc[trainInfo.sig_id,:]
            cols = cmap_train.columns.values
            cmap_val = cmap.loc[valInfo.sig_id,:]
            N = len(cmap_train)
            r_train = np.corrcoef(cmap_train.T)
            r_train = pd.DataFrame(r_train,index=cols,columns=cols)
            r_train = r_train.loc[genes_1,genes_2]
            r_train = r_train.reset_index(drop=False).melt(id_vars=['index'],var_name='index_2',value_name='r').rename(columns={'index':'gene_1','index_2':'gene_2'})
            r_train = r_train[r_train.gene_1!=r_train.gene_2]
            r_train['fold'] = fold_id
            r_train['set'] = 'train'
            r_train['cell'] = cell
            r_train['iteration'] = j
            ## repeat for validation
            r_val = np.corrcoef(cmap_val.T)
            r_val = pd.DataFrame(r_val,index=cols,columns=cols)
            r_val = r_val.loc[genes_1,genes_2]
            r_val = r_val.reset_index(drop=False).melt(id_vars=['index'],var_name='index_2',value_name='r').rename(columns={'index':'gene_1','index_2':'gene_2'})
            r_val = r_val[r_val.gene_1!=r_val.gene_2]
            r_val['fold'] = fold_id
            r_val['set'] = 'validation'
            r_val['cell'] = cell
            r_val['iteration'] = j
            # combine
            r_fold = pd.concat([r_train,r_val])
            r_all = pd.concat([r_all,r_fold])
r_all.to_csv(output_dir+'InitialCorrelationOfGenesSubsets.csv')