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
parser = argparse.ArgumentParser(description="Run comparison of TRANSACT with GPU implementation VS CPU implementation using random data.")
parser.add_argument('--distribution', type=str, 
                    choices=['normal', 'uniform', 'xavier', 'exponential', 'dirichlet', 
                            'lognormal', 'poisson', 'gamma'], 
                    default='normal', 
                    help='Distribution to generate random data from')
parser.add_argument('--output_dir', type=str, help='Directory to save output results.',
                    default='../results/GPU_vs_CPU_random/')
parser.add_argument('--random_iterations', type=int, help='Number of random iterations per combination.',
                    default=30)

args = parser.parse_args()
distribution = args.distribution
output_dir = args.output_dir
random_iterations = args.random_iterations

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print2log(f'Using device: {device}')

# Initialize environment and seeds for reproducibility
seed_everything(42)

print2log(f'Using distribution: {distribution}')
print2log(f'Random iterations per combination: {random_iterations}')

def generate_random_matrix(n_samples, n_features, dist_type='normal'):
    """
    Generate random matrix from specified distribution
    
    Args:
        n_samples: Number of samples (rows)
        n_features: Number of features (columns)
        dist_type: Type of distribution
    
    Returns:
        torch tensor of shape (n_samples, n_features)
    """
    if dist_type == 'normal':
        # Standard normal distribution
        matrix = np.random.randn(n_samples, n_features).astype(np.float32)
    
    elif dist_type == 'uniform':
        # Uniform distribution between -1 and 1
        matrix = np.random.uniform(-1, 1, (n_samples, n_features)).astype(np.float32)
    
    elif dist_type == 'xavier':
        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (n_samples + n_features))
        matrix = np.random.uniform(-limit, limit, (n_samples, n_features)).astype(np.float32)
    
    elif dist_type == 'exponential':
        # Exponential distribution (lambda=1) - skewed positive values
        matrix = np.random.exponential(scale=1.0, size=(n_samples, n_features)).astype(np.float32)
    
    elif dist_type == 'dirichlet':
        # Dirichlet distribution - compositional data (rows sum to 1)
        alpha = np.ones(n_features)  # symmetric Dirichlet
        matrix = np.array([np.random.dirichlet(alpha) for _ in range(n_samples)]).astype(np.float32)
    
    elif dist_type == 'lognormal':
        # Log-normal distribution - positive values, heavy-tailed
        matrix = np.random.lognormal(mean=0, sigma=1, size=(n_samples, n_features)).astype(np.float32)
    
    elif dist_type == 'poisson':
        # Poisson distribution (lambda=5) - count-like positive integers
        matrix = np.random.poisson(lam=5, size=(n_samples, n_features)).astype(np.float32)
    
    elif dist_type == 'gamma':
        # Gamma distribution - positive continuous values
        matrix = np.random.gamma(shape=2.0, scale=2.0, size=(n_samples, n_features)).astype(np.float32)
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    
    return torch.tensor(matrix).float()

# Define sample sizes and feature spaces to test
sample_sizes = [32, 64, 128, 256, 512, 1024]
feature_spaces = [64, 128, 256, 512, 1024, 2048]

# Storage for results
all_cpu_times = []
all_gpu_times = []
all_results = []

# Dataframes for pearson correlations
df_pear_1 = pd.DataFrame()
df_pear_2 = pd.DataFrame()
df_pear_1_val = pd.DataFrame()
df_pear_2_val = pd.DataFrame()

print2log(f"\n{'='*80}")
print2log(f"Starting experiments with {len(sample_sizes)} sample sizes and {len(feature_spaces)} feature spaces")
print2log(f"Total combinations: {len(sample_sizes) * len(feature_spaces)}")
print2log(f"Note: GPU speedup is typically only visible for larger matrices (>256 samples)")
print2log(f"{'='*80}\n")

# Loop over all combinations
combination_count = 0
total_combinations = len(sample_sizes) * len(feature_spaces)

for feature_size in feature_spaces:
    for sample_size in sample_sizes:
        combination_count += 1
        print2log(f"\n{'='*80}")
        print2log(f"Combination {combination_count}/{total_combinations}")
        print2log(f"Feature space: {feature_size}, Sample size: {sample_size}")
        print2log(f"{'='*80}")
        
        # Storage for this combination
        pearsons_1 = np.zeros((random_iterations, 30))
        pearsons_2 = np.zeros((random_iterations, 30))
        pearsons_1_val = np.zeros((random_iterations, 30))
        pearsons_2_val = np.zeros((random_iterations, 30))
        
        cpu_times_combo = []
        gpu_times_combo = []
        
        # Run random_iterations experiments for this combination
        for iteration in range(random_iterations):
            if iteration % 5 == 0:
                print2log(f"  Iteration {iteration+1}/{random_iterations}")
            
            # Generate 4 random matrices for training and validation
            # Generate on CPU first to ensure fair comparison
            Xtrain_1_cpu = generate_random_matrix(sample_size, feature_size, distribution)
            Xtrain_2_cpu = generate_random_matrix(sample_size, feature_size, distribution)
            X_1_val_cpu = generate_random_matrix(sample_size, feature_size, distribution)
            X_2_val_cpu = generate_random_matrix(sample_size, feature_size, distribution)
            
            # Move to GPU
            Xtrain_1 = Xtrain_1_cpu.to(device)
            Xtrain_2 = Xtrain_2_cpu.to(device)
            X_1_val = X_1_val_cpu.to(device)
            X_2_val = X_2_val_cpu.to(device)
            
            # ============================================
            # GPU Implementation
            # ============================================
            # CRITICAL: Ensure all previous GPU operations are complete
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time_gpu_align = time.perf_counter()
            
            X_2_aligned_gpu, X_1_aligned_gpu, _, initial_alignment_model_gpu = transact_align_gpu(
                Xtrain_2,  # source
                Xtrain_1,  # target
                n_src_pcs=min(75, min(sample_size, feature_size) - 1),
                n_tgt_pcs=min(75, min(sample_size, feature_size) - 1),
                n_pv=min(30, min(sample_size, feature_size) - 1),
                kernel='rbf',
                gamma=5e-4,
                device=device)
            
            X_2_transformed_val_gpu = transact_transform_gpu(X_2_val, initial_alignment_model_gpu, space='source')
            X_1_transformed_val_gpu = transact_transform_gpu(X_1_val, initial_alignment_model_gpu, space='target')
            
            # CRITICAL: Ensure all GPU operations are complete before stopping timer
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time_gpu_align = time.perf_counter()
            time_gpu_align = end_time_gpu_align - start_time_gpu_align
            
            # ============================================
            # CPU Implementation
            # ============================================
            # Convert to numpy (already on CPU)
            x2train_cpu = Xtrain_2_cpu.numpy()
            x1train_cpu = Xtrain_1_cpu.numpy()
            x2_val_cpu = X_2_val_cpu.numpy()
            x1_val_cpu = X_1_val_cpu.numpy()
            
            start_time_cpu_align = time.perf_counter()
            
            X_2_aligned_cpu, X_1_aligned_cpu, _, initial_alignment_model_cpu = transact_align(
                x2train_cpu,  # source
                x1train_cpu,  # target
                n_src_pcs=min(75, min(sample_size, feature_size) - 1),
                n_tgt_pcs=min(75, min(sample_size, feature_size) - 1),
                n_pv=min(30, min(sample_size, feature_size) - 1),
                kernel='rbf',
                gamma=5e-4)
            
            X_2_transformed_val_cpu = transact_transform(x2_val_cpu, initial_alignment_model_cpu, space='source')
            X_1_transformed_val_cpu = transact_transform(x1_val_cpu, initial_alignment_model_cpu, space='target')
            
            end_time_cpu_align = time.perf_counter()
            time_cpu_align = end_time_cpu_align - start_time_cpu_align
            
            # Store timing results
            cpu_times_combo.append(time_cpu_align)
            gpu_times_combo.append(time_gpu_align)
            all_cpu_times.append(time_cpu_align)
            all_gpu_times.append(time_gpu_align)
            
            # ============================================
            # Calculate Pearson Correlations
            # ============================================

            # First match the matrices in case there is a re-ordering of the PVs
            X_1_aligned_gpu_matched,_ = match_pvs(X_1_aligned_cpu, X_1_aligned_gpu.detach().cpu().numpy())
            X_2_aligned_gpu_matched,_ = match_pvs(X_2_aligned_cpu, X_2_aligned_gpu.detach().cpu().numpy())
            # Training alignments
            p1 = pearson_r(torch.tensor(X_1_aligned_cpu).to(device), torch.tensor(X_1_aligned_gpu_matched).to(device)).abs().detach().cpu().numpy()
            p2 = pearson_r(torch.tensor(X_2_aligned_cpu).to(device), torch.tensor(X_2_aligned_gpu_matched).to(device)).abs().detach().cpu().numpy()
            # p1 = pearson_r(torch.tensor(X_1_aligned_cpu).to(device), X_1_aligned_gpu).abs().detach().cpu().numpy()
            # p2 = pearson_r(torch.tensor(X_2_aligned_cpu).to(device), X_2_aligned_gpu).abs().detach().cpu().numpy()
            
            # Validation transformations
            p1_val = pearson_r(torch.tensor(X_1_transformed_val_cpu).to(device), X_1_transformed_val_gpu).abs().detach().cpu().numpy()
            p2_val = pearson_r(torch.tensor(X_2_transformed_val_cpu).to(device), X_2_transformed_val_gpu).abs().detach().cpu().numpy()
            
            # Store pearson correlations
            pearsons_1[iteration, :] = p1
            pearsons_2[iteration, :] = p2
            pearsons_1_val[iteration, :] = p1_val
            pearsons_2_val[iteration, :] = p2_val
            
            # Store detailed results
            all_results.append({
                'feature_size': feature_size,
                'sample_size': sample_size,
                'iteration': iteration,
                'cpu_time': time_cpu_align,
                'gpu_time': time_gpu_align,
                'speedup': time_cpu_align / time_gpu_align,
                'min_pearson_1': p1.min(),
                'mean_pearson_1': p1.mean(),
                'min_pearson_2': p2.min(),
                'mean_pearson_2': p2.mean(),
                'min_pearson_1_val': p1_val.min(),
                'mean_pearson_1_val': p1_val.mean(),
                'min_pearson_2_val': p2_val.min(),
                'mean_pearson_2_val': p2_val.mean()
            })
        
        # ============================================
        # Process results for this combination
        # ============================================
        avg_cpu_time = np.mean(cpu_times_combo)
        avg_gpu_time = np.mean(gpu_times_combo)
        avg_speedup = avg_cpu_time / avg_gpu_time
        
        print2log(f"\n  Results for Feature={feature_size}, Sample={sample_size}:")
        print2log(f"    Avg CPU time: {avg_cpu_time:.4f}s")
        print2log(f"    Avg GPU time: {avg_gpu_time:.4f}s")
        print2log(f"    Avg Speedup: {avg_speedup:.2f}x")
        if avg_speedup < 1.0:
            print2log(f"    ⚠️  GPU is SLOWER (overhead dominates for small matrices)")
        print2log(f"    Min Pearson X_1: {pearsons_1.min():.6f}")
        print2log(f"    Min Pearson X_2: {pearsons_2.min():.6f}")
        print2log(f"    Min Pearson X_1_val: {pearsons_1_val.min():.6f}")
        print2log(f"    Min Pearson X_2_val: {pearsons_2_val.min():.6f}")
        
        # Convert pearson correlations to dataframes
        for pearson_data, df_storage, var_name in [
            (pearsons_1, df_pear_1, 'X_1'),
            (pearsons_2, df_pear_2, 'X_2'),
            (pearsons_1_val, df_pear_1_val, 'X_1_val'),
            (pearsons_2_val, df_pear_2_val, 'X_2_val')
        ]:
            df_temp = pd.DataFrame(pearson_data, index=range(random_iterations))
            df_temp.columns = ['pv_' + str(i) for i in np.arange(1, 31)]
            df_temp = pd.melt(df_temp.reset_index(), id_vars=['index'])
            df_temp.rename(columns={'index': 'iteration'}, inplace=True)
            df_temp['sample_size'] = sample_size
            df_temp['feature_size'] = feature_size
            df_temp['variable_name'] = var_name
            
            if var_name == 'X_1':
                df_pear_1 = pd.concat([df_pear_1, df_temp], axis=0)
            elif var_name == 'X_2':
                df_pear_2 = pd.concat([df_pear_2, df_temp], axis=0)
            elif var_name == 'X_1_val':
                df_pear_1_val = pd.concat([df_pear_1_val, df_temp], axis=0)
            elif var_name == 'X_2_val':
                df_pear_2_val = pd.concat([df_pear_2_val, df_temp], axis=0)

# ============================================
# Save all results
# ============================================
print2log(f"\n{'='*80}")
print2log("SAVING RESULTS")
print2log(f"{'='*80}\n")

# Save pearson correlations
output_prefix = f"{distribution}_"
df_pear_1.to_csv(os.path.join(output_dir, f'{output_prefix}pearsons_1.csv'), index=False)
df_pear_2.to_csv(os.path.join(output_dir, f'{output_prefix}pearsons_2.csv'), index=False)
df_pear_1_val.to_csv(os.path.join(output_dir, f'{output_prefix}pearsons_1_val.csv'), index=False)
df_pear_2_val.to_csv(os.path.join(output_dir, f'{output_prefix}pearsons_2_val.csv'), index=False)

print2log(f"Saved Pearson correlation files with prefix: {output_prefix}")

# Save detailed results
df_results = pd.DataFrame(all_results)
df_results.to_csv(os.path.join(output_dir, f'{output_prefix}detailed_results.csv'), index=False)
print2log(f"Saved detailed results")

# Save timing summary
timing_summary = []
for feature_size in feature_spaces:
    for sample_size in sample_sizes:
        mask = (df_results['feature_size'] == feature_size) & (df_results['sample_size'] == sample_size)
        subset = df_results[mask]
        timing_summary.append({
            'feature_size': feature_size,
            'sample_size': sample_size,
            'avg_cpu_time': subset['cpu_time'].mean(),
            'std_cpu_time': subset['cpu_time'].std(),
            'avg_gpu_time': subset['gpu_time'].mean(),
            'std_gpu_time': subset['gpu_time'].std(),
            'avg_speedup': subset['speedup'].mean(),
            'std_speedup': subset['speedup'].std()
        })

df_timing = pd.DataFrame(timing_summary)
df_timing.to_csv(os.path.join(output_dir, f'{output_prefix}timing_summary.csv'), index=False)
print2log(f"Saved timing summary")

# ============================================
# Print final summary statistics
# ============================================
print2log(f"\n{'='*80}")
print2log("FINAL SUMMARY")
print2log(f"{'='*80}\n")

print2log(f"Distribution: {distribution}")
print2log(f"Total experiments run: {len(all_results)}")
print2log(f"Feature spaces tested: {feature_spaces}")
print2log(f"Sample sizes tested: {sample_sizes}")
print2log(f"Iterations per combination: {random_iterations}\n")

print2log(f"Overall Average CPU time: {np.mean(all_cpu_times):.4f}s ± {np.std(all_cpu_times):.4f}s")
print2log(f"Overall Average GPU time: {np.mean(all_gpu_times):.4f}s ± {np.std(all_gpu_times):.4f}s")
overall_speedup = np.mean(np.array(all_cpu_times)/np.array(all_gpu_times))
print2log(f"Overall Average Speedup: {overall_speedup:.2f}x")

if overall_speedup < 1.0:
    print2log(f"⚠️  WARNING: GPU appears slower overall. This is expected for small matrices.")
    print2log(f"   GPU overhead (memory transfer, kernel launch) dominates for small data.")
    print2log(f"   Your 700×978 data should show speedup. Try larger sample sizes (>256).")

print2log(f"\nPearson Correlation Statistics:")
print2log(f"  X_1 (train) - Mean: {df_pear_1['value'].mean():.6f}, Min: {df_pear_1['value'].min():.6f}")
print2log(f"  X_2 (train) - Mean: {df_pear_2['value'].mean():.6f}, Min: {df_pear_2['value'].min():.6f}")
print2log(f"  X_1 (val)   - Mean: {df_pear_1_val['value'].mean():.6f}, Min: {df_pear_1_val['value'].min():.6f}")
print2log(f"  X_2 (val)   - Mean: {df_pear_2_val['value'].mean():.6f}, Min: {df_pear_2_val['value'].min():.6f}")

# Print timing by feature size
print2log(f"\n{'='*80}")
print2log("TIMING BY FEATURE SIZE")
print2log(f"{'='*80}\n")
for feature_size in feature_spaces:
    subset = df_timing[df_timing['feature_size'] == feature_size]
    avg_speedup = subset['avg_speedup'].mean()
    status = "✓ GPU faster" if avg_speedup > 1.0 else "⚠️ GPU slower"
    print2log(f"Feature size {feature_size}: Avg speedup = {avg_speedup:.2f}x  {status}")

# Print timing by sample size
print2log(f"\n{'='*80}")
print2log("TIMING BY SAMPLE SIZE")
print2log(f"{'='*80}\n")
for sample_size in sample_sizes:
    subset = df_timing[df_timing['sample_size'] == sample_size]
    avg_speedup = subset['avg_speedup'].mean()
    status = "✓ GPU faster" if avg_speedup > 1.0 else "⚠️ GPU slower"
    print2log(f"Sample size {sample_size}: Avg speedup = {avg_speedup:.2f}x  {status}")

print2log(f"\n{'='*80}")
print2log("ALL RESULTS SAVED TO: " + output_dir)
print2log(f"{'='*80}\n")