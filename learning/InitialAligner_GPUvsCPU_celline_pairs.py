import torch
import numpy as np
import warnings
from utility import *
from evaluationUtils import pearson_r
import time
import argparse
from utility import *
from transact_utility_gpu import *
from pathlib import Path
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

def match_pvs(X_cpu, X_gpu):
    """Match principal vectors between CPU and GPU implementations."""
    n_pv = X_cpu.shape[1]
    corr_matrix = np.zeros((n_pv, n_pv))
    
    # Compute absolute correlations (handles sign flips)
    for i in range(n_pv):
        for j in range(n_pv):
            corr_matrix[i, j] = np.abs(pearsonr(X_cpu[:, i], X_gpu[:, j])[0])
    
    # Use Hungarian algorithm to find best matching
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-corr_matrix)  # Maximize correlation
    
    # Reorder GPU to match CPU
    X_gpu_matched = X_gpu[:, col_ind]
    
    # Fix signs
    for i in range(n_pv):
        if pearsonr(X_cpu[:, i], X_gpu_matched[:, i])[0] < 0:
            X_gpu_matched[:, i] *= -1
    
    return X_gpu_matched, corr_matrix[row_ind, col_ind]

# Initialize argparse
parser = argparse.ArgumentParser(description="Run comparison of TRANSACT with GPU implementation VS CPU implementation.")
# Data and output paths
parser.add_argument('--folders', metavar='N', type=str, nargs='*', help='folders with paired datasets',
                    default=['A375_HT29', 'A375_PC3', 'HA1E_VCAP', 'HT29_MCF7', 'HT29_PC3', 'MCF7_HA1E', 'MCF7_PC3', 'PC3_HA1E'])
parser.add_argument('--data_root', type=str, help='Root directory for preprocessed data.',default='../preprocessing/preprocessed_data/CellPairs/')
parser.add_argument('--cmap_file', type=str, help='Path to the CMAP CSV file.',default='../preprocessing/preprocessed_data/CellPairs/drug_landmarks.csv')
parser.add_argument('--output_dir', type=str, help='Directory to save output results.',default='../results/GPU_vs_CPU/')


args = parser.parse_args()
folders = args.folders
output_dir = args.output_dir

Path(output_dir).mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print2log(f'Using device: {device}')

# Initialize environment and seeds for reproducibility
seed_everything(42)

# Read data
cmap = pd.read_csv(args.cmap_file, index_col=0)
genes = cmap.columns.values
gene_size = len(cmap.columns)
samples = cmap.index.values
fold_id = 1 # just select on and it will give the total dataset
all_cpu_timpes = []
all_gpu_timpes = []
pearsons_1 = np.zeros((len(folders),30))
pearsons_2 = np.zeros((len(folders),30))
pearsons_2_val = np.zeros((len(folders),30))
pearsons_1_val = np.zeros((len(folders),30))
for i, folder in enumerate(folders):
    # Extract dataset names from the folder name
    dataset1, dataset2 = folder.split('_')
    folder_path = os.path.join(args.data_root, folder)
    largest_sample_len = find_largest_sample_len(folder_path)
    print2log(f'Processing folder: {folder} with sample_len: {largest_sample_len}')

    trainInfo_paired = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'train_paired_{fold_id}.csv'), index_col=None)
    trainInfo_1 = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'train_{dataset1}_{fold_id}.csv'), index_col=None)
    trainInfo_2 = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'train_{dataset2}_{fold_id}.csv'), index_col=None)
    valInfo_paired = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'val_paired_{fold_id}.csv'), index_col=None)
    valInfo_1 = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'val_{dataset1}_{fold_id}.csv'), index_col=None)
    valInfo_2 = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'val_{dataset2}_{fold_id}.csv'), index_col=None)

    # from the above metadata get x_train, y_train, x_val, y_val
    Xtrain_1 = torch.tensor(np.concatenate((cmap.loc[trainInfo_paired['sig_id.x']].values,
                                                 cmap.loc[trainInfo_1.sig_id].values))).float().to(device)
    Xtrain_2 = torch.tensor(np.concatenate((cmap.loc[trainInfo_paired['sig_id.y']].values,
                                           cmap.loc[trainInfo_2.sig_id].values))).float().to(device)
    # x for validation
    X_1_val = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,
                                                 cmap.loc[valInfo_1.sig_id].values))).float().to(device)
    X_2_val = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,
                                                 cmap.loc[valInfo_2.sig_id].values))).float().to(device)
    if Xtrain_1.shape[0] > Xtrain_2.shape[0]:
        Xtrain_1 = Xtrain_1[:Xtrain_2.shape[0],:]
    elif Xtrain_2.shape[0] > Xtrain_1.shape[0]:
        Xtrain_2 = Xtrain_2[:Xtrain_1.shape[0],:]
    torch.cuda.synchronize()

    # X_1 = torch.randn(771, 978, device=device)  # Target
    # X_2 = torch.randn(771, 978, device=device)  # Source
    
    X_2_aligned_gpu, X_1_aligned_gpu, tau_opt, initial_alignment_model = transact_align_gpu(
            Xtrain_2,  # source
            Xtrain_1,  # target
            n_src_pcs=75,
            n_tgt_pcs=75,
            n_pv=30,
            kernel='rbf',
            gamma=5e-4,
            device=device)
    torch.cuda.synchronize()
    start_time_gpu_align = time.perf_counter()
    X_2_aligned_gpu, X_1_aligned_gpu, _, initial_alignment_model = transact_align_gpu(
            Xtrain_2,  # source
            Xtrain_1,  # target
            n_src_pcs=75,
            n_tgt_pcs=75,
            n_pv=30,
            kernel='rbf',
            gamma=5e-4,
            device=device)

    X_2_transformed_val_gpu = transact_transform_gpu(X_2_val, initial_alignment_model, space='source')
    X_1_transformed_val_gpu = transact_transform_gpu(X_1_val, initial_alignment_model, space='target')

    end_time_gpu_align = time.perf_counter()
    time_gpu_align = end_time_gpu_align - start_time_gpu_align

    print(f"Repeat with numpy implementation from utility.py")
    x2train_cpu = Xtrain_2.cpu().numpy()
    x1train_cpu = Xtrain_1.cpu().numpy()
    x2_val_cpu = X_2_val.cpu().numpy()
    x1_val_cpu = X_1_val.cpu().numpy()
    start_time_cpu_align = time.perf_counter()
    X_2_aligned, X_1_aligned, _, initial_alignment_model = transact_align(
        x2train_cpu,  # source
        x1train_cpu,  # target
        n_src_pcs=75,
        n_tgt_pcs=75,
        n_pv=30,
        kernel='rbf',
        gamma=5e-4)

    X_2_transformed_val_cpu = transact_transform(x2_val_cpu, initial_alignment_model, space='source')
    X_1_transformed_val_cpu = transact_transform(x1_val_cpu, initial_alignment_model, space='target')
    end_time_cpu_align = time.perf_counter()
    time_cpu_align = end_time_cpu_align - start_time_cpu_align

    all_cpu_timpes.append(time_cpu_align)
    all_gpu_timpes.append(time_gpu_align)

    # First match the matrices in case there is a re-ordering of the PVs
    X_1_aligned_gpu_matched,_ = match_pvs(X_1_aligned, X_1_aligned_gpu.detach().cpu().numpy())
    X_2_aligned_gpu_matched,_ = match_pvs(X_2_aligned, X_2_aligned_gpu.detach().cpu().numpy())
    p2 = pearson_r(torch.tensor(X_2_aligned).to(device), torch.tensor(X_2_aligned_gpu_matched).to(device)).abs().detach().cpu().numpy()
    p1 = pearson_r(torch.tensor(X_1_aligned).to(device), torch.tensor(X_1_aligned_gpu_matched).to(device)).abs().detach().cpu().numpy()
    p2_new = pearson_r(torch.tensor(X_2_transformed_val_cpu).to(device), X_2_transformed_val_gpu).abs().detach().cpu().numpy()
    p1_new = pearson_r(torch.tensor(X_1_transformed_val_cpu).to(device), X_1_transformed_val_gpu).abs().detach().cpu().numpy()

    print2log(f"Min pearson for X_1_aligned: {p1.min()}")
    print2log(f"Min pearson for X_2_aligned: {p2.min()}")
    print2log(f"Min pearson for X_2_transformed: {p2_new.min()}")
    print2log(f"Min pearson for X_1_transformed: {p1_new.min()}")

    pearsons_1[i,:]=p1
    pearsons_2[i,:]=p2
    pearsons_2_val[i,:]=p2_new
    pearsons_1_val[i,:]=p1_new
    
    print2log('Completely finished pair:'+folder)

pearsons_1 = pd.DataFrame(pearsons_1,index=folders)
pearsons_1.columns = ['x_'+str(i) for i in np.arange(1,31)]
pearsons_1 = pd.melt(pearsons_1.reset_index(), id_vars=['index'])
pearsons_1.rename(columns={'index':'folder'},inplace=True)
pearsons_1.to_csv(output_dir+'cellLinePairs_pearsons_1.csv')

pearsons_2 = pd.DataFrame(pearsons_2,index=folders)
pearsons_2.columns = ['x_'+str(i) for i in np.arange(1,31)]
pearsons_2 = pd.melt(pearsons_2.reset_index(), id_vars=['index'])
pearsons_2.rename(columns={'index':'folder'},inplace=True)
pearsons_2.to_csv(output_dir+'cellLinePairs_pearsons_2.csv')

pearsons_2_val = pd.DataFrame(pearsons_2_val,index=folders)
pearsons_2_val.columns = ['x_'+str(i) for i in np.arange(1,31)]
pearson_val = pd.melt(pearsons_2_val.reset_index(), id_vars=['index'])
pearson_val.rename(columns={'index':'folder'},inplace=True)
pearson_val.to_csv(output_dir+'cellLinePairs_pearsons_2_val.csv')

pearsons_1_val = pd.DataFrame(pearsons_1_val,index=folders)
pearsons_1_val.columns = ['x_'+str(i) for i in np.arange(1,31)]
pearson_val = pd.melt(pearsons_1_val.reset_index(), id_vars=['index'])
pearson_val.rename(columns={'index':'folder'},inplace=True)
pearson_val.to_csv(output_dir+'cellLinePairs_pearsons_1_val.csv')

df = pd.DataFrame({'folder':folders,'GPU':all_gpu_timpes,'CPU':all_cpu_timpes})
df.to_csv(output_dir+'cellLinePairs_time.csv')

print2log(f"\nAverage GPU time over all pairs: {np.mean(all_gpu_timpes)}")
print2log(f"Average CPU time over all pairs: {np.mean(all_cpu_timpes)}")
print2log(f"GPU vs CPU speedup: ~{round(np.mean(np.array(all_cpu_timpes)/np.array(all_gpu_timpes)),2)}x")