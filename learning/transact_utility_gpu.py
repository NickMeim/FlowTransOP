import torch
import numpy as np
import warnings
from utility import *
from evaluationUtils import pearson_r
import time
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 0. PyTorch StandardScaler (GPU-compatible)
# ------------------------------------------------------------------
class TorchStandardScaler:
    """StandardScaler implemented in PyTorch for GPU compatibility."""
    
    def __init__(self, eps=1e-7):
        self.eps = eps
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X):
        """Compute mean and std for each feature."""
        self.mean_ = torch.mean(X, dim=0, keepdim=True)
        # Match sklearn: ddof=0 -> unbiased=False
        self.std_ = torch.std(X, dim=0, keepdim=True, unbiased=False)
        self.std_ = torch.clamp(self.std_, min=self.eps)
        return self
    
    def transform(self, X):
        """Standardize features."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Reverse the standardization."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return X * self.std_ + self.mean_

# ------------------------------------------------------------------
# 1. GPU Kernel Functions
# ------------------------------------------------------------------
def rbf_kernel_torch(X, Y, gamma):
    """
    Compute RBF kernel on GPU: K(x,y) = exp(-gamma * ||x-y||^2)
    X: (n, d), Y: (m, d) -> K: (n, m)
    """
    X_norm = (X ** 2).sum(1).view(-1, 1)
    Y_norm = (Y ** 2).sum(1).view(1, -1)
    dists = X_norm + Y_norm - 2.0 * torch.mm(X, Y.t())
    return torch.exp(-gamma * dists)

def linear_kernel_torch(X, Y):
    """Linear kernel on GPU: K(x,y) = x^T y"""
    return torch.mm(X, Y.t())

# ------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------
def center_kernel_torch(K):
    """Double-centres a kernel matrix on GPU."""
    n = K.shape[0]
    one = torch.ones(n, n, device=K.device, dtype=K.dtype) / n
    return K - one @ K - K @ one + one @ K @ one

def kernel_pca_torch(Kc, n_components, eps=1e-12):
    """
    Kernel-PCA on GPU using centered kernel.
    Returns alpha (n_components × n_samples) with orthonormal rows.
    """
    eigvals, eigvecs = torch.linalg.eigh(Kc)
    
    idx = torch.argsort(eigvals, descending=True)[:n_components]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    eigvals = eigvals.unsqueeze(1)  # (n_components, 1)
    eigvals = torch.clamp(eigvals, min=0.0)
    
    alpha = eigvecs.t() / torch.sqrt(eigvals + eps)
    return alpha

# ------------------------------------------------------------------
# 3. KS Test on GPU (vectorized)
# ------------------------------------------------------------------
def ks_statistic_torch(x, y):
    """Numerically stable KS statistic matching scipy."""
    n, m = len(x), len(y)
    
    # Sort both samples
    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)
    
    # Combine and sort
    combined = torch.cat([x_sorted, y_sorted])
    _, sort_idx = torch.sort(combined)
    
    # Track which sample each value came from
    source = torch.cat([torch.zeros(n, device=x.device), 
                       torch.ones(m, device=y.device)])[sort_idx]
    
    # Compute ECDF differences at each point
    cdf_x = torch.cumsum(source == 0, dim=0).float() / n
    cdf_y = torch.cumsum(source == 1, dim=0).float() / m
    
    # KS statistic (add small epsilon for stability)
    stat = torch.max(torch.abs(cdf_x - cdf_y)) + 1e-10 * torch.randn(1, device=x.device).abs()
    return stat
# def ks_statistic_torch(x, y):
#     """
#     Compute Kolmogorov-Smirnov statistic on GPU.
#     x: (n_samples,)
#     y: (m_samples,)
#     Returns KS statistic
#     """
#     n, m = len(x), len(y)
    
#     # Sort both samples
#     x_sorted, _ = torch.sort(x)
#     y_sorted, _ = torch.sort(y)
    
#     # Combine all unique values
#     combined = torch.cat([x_sorted, y_sorted])
#     sorted_combined, _ = torch.sort(combined)
    
#     # Compute ECDFs at combined points
#     # For x: count how many values in x are <= each point
#     cdf_x = torch.searchsorted(x_sorted, sorted_combined).float() / n
#     cdf_y = torch.searchsorted(y_sorted, sorted_combined).float() / m
    
#     # KS statistic
#     stat = torch.max(torch.abs(cdf_x - cdf_y))
#     return stat

# ------------------------------------------------------------------
# 4. Main TRANSACT Alignment (Fully GPU version)
# ------------------------------------------------------------------
def transact_align_gpu(
        Xs, Xt,
        n_src_pcs=75, n_tgt_pcs=75, n_pv=30,
        kernel='rbf', gamma=5e-4,
        cross_kernel=None,
        grid=None,
        device='cuda'
    ):
    """
    TRANSACT alignment fully on GPU using PyTorch.
    
    Parameters
    ----------
    Xs : torch.Tensor or np.ndarray, shape (ns, ps)
        Source domain data
    Xt : torch.Tensor or np.ndarray, shape (nt, pt)
        Target domain data
    n_src_pcs : int
        Number of source principal components
    n_tgt_pcs : int
        Number of target principal components
    n_pv : int
        Number of principal vectors
    kernel : str
        'rbf' or 'linear'
    gamma : float
        RBF kernel bandwidth
    cross_kernel : callable, optional
        Custom cross-domain kernel function
    grid : array-like, optional
        Grid for tau search
    device : str
        'cuda' or 'cpu'
        
    Returns
    -------
    Zs : torch.Tensor, shape (ns, n_pv)
        Source consensus scores
    Zt : torch.Tensor, shape (nt, n_pv)
        Target consensus scores
    tau_opt : torch.Tensor, shape (n_pv,)
        Optimal interpolation parameters
    model : dict
        Model dictionary for transformation
    """
    if grid is None:
        grid = np.linspace(0., 1., 21)
    
    # Convert to torch tensors if needed
    if not torch.is_tensor(Xs):
        Xs = torch.from_numpy(Xs).float()
    if not torch.is_tensor(Xt):
        Xt = torch.from_numpy(Xt).float()
    
    # Move to device
    Xs = Xs.to(device)
    Xt = Xt.to(device)
    
    # ---- 4.1 Standardization (fully on GPU) ----
    Xs_scaler = TorchStandardScaler().fit(Xs)
    Xt_scaler = TorchStandardScaler().fit(Xt)
    Xs = Xs_scaler.transform(Xs)
    Xt = Xt_scaler.transform(Xt)
    
    # ---- 4.2 Build kernels ----
    if kernel == 'linear':
        Ks = linear_kernel_torch(Xs, Xs)
        Kt = linear_kernel_torch(Xt, Xt)
        Kst = linear_kernel_torch(Xs, Xt) if cross_kernel is None else cross_kernel(Xs, Xt)
    elif kernel == 'rbf':
        Ks = rbf_kernel_torch(Xs, Xs, gamma)
        Kt = rbf_kernel_torch(Xt, Xt, gamma)
        if cross_kernel is None:
            common = min(Xs.shape[1], Xt.shape[1])
            Kst = rbf_kernel_torch(Xs[:, :common], Xt[:, :common], gamma)
        else:
            Kst = cross_kernel(Xs, Xt)
    else:
        raise ValueError("kernel must be 'linear' or 'rbf'")
    
    # ---- 4.3 Center kernels ----
    Ks_c = center_kernel_torch(Ks)
    Kt_c = center_kernel_torch(Kt)
    
    ns, nt = Xs.shape[0], Xt.shape[0]
    Cns = torch.eye(ns, device=device, dtype=Xs.dtype) - 1.0 / ns
    Cnt = torch.eye(nt, device=device, dtype=Xt.dtype) - 1.0 / nt
    Kst_c = Cns @ Kst @ Cnt
    
    # ---- 4.4 Kernel PCA ----
    alpha_s = kernel_pca_torch(Ks_c, n_src_pcs)  # (n_src_pcs, ns)
    alpha_t = kernel_pca_torch(Kt_c, n_tgt_pcs)  # (n_tgt_pcs, nt)
    
    # ---- 4.5 Cosine similarity matrix ----
    MK = alpha_s @ Kst_c @ alpha_t.t()  # (n_src_pcs, n_tgt_pcs)
    
    # ---- 4.6 SVD for Principal Vectors ----
    U, Sigma, Vt = torch.linalg.svd(MK, full_matrices=False)
    
    beta_s = U[:, :n_pv]
    beta_t = Vt.t()[:, :n_pv]
    
    rho_s = beta_s.t() @ alpha_s  # (n_pv, ns)
    rho_t = beta_t.t() @ alpha_t  # (n_pv, nt)
    
    theta = torch.acos(torch.clamp(Sigma[:n_pv], 0, 1))
    
    # ---- 4.7 Evaluate PVs on samples ----
    Ss = Ks_c @ rho_s.t()      # (ns, n_pv)
    Ts = Kst_c @ rho_t.t()     # (ns, n_pv)
    St = Kst_c.t() @ rho_s.t()  # (nt, n_pv)
    Tt = Kt_c @ rho_t.t()      # (nt, n_pv)
    
    # ---- 4.8 KS search for optimal tau (per PV) ----
    Zs_list, Zt_list, tau_opt_list = [], [], []
    
    for q in range(n_pv):
        s_q = Ss[:, q]
        t_q = Tt[:, q]
        st_q = St[:, q]
        ts_q = Ts[:, q]
        
        best_tau = 0.5
        best_stat = float('inf')
        best_proj_s = None
        best_proj_t = None
        
        for tau in grid:
            tau_t = torch.tensor(tau, device=device, dtype=Xs.dtype)
            
            # Geodesic interpolation
            if torch.sin(theta[q]) < 1e-8:
                GAMMA = 1 - tau_t
                KSI = tau_t
            else:
                GAMMA = torch.sin((1 - tau_t) * theta[q]) / torch.sin(theta[q])
                KSI = torch.sin(tau_t * theta[q]) / torch.sin(theta[q])
            
            proj_s = GAMMA * s_q + KSI * st_q
            proj_t = GAMMA * ts_q + KSI * t_q
            
            # Compute KS statistic
            stat = ks_statistic_torch(proj_s, proj_t).item()
            
            if stat < best_stat:
                best_stat = stat
                best_tau = tau
                best_proj_s = proj_s.clone()
                best_proj_t = proj_t.clone()
        
        Zs_list.append(best_proj_s)
        Zt_list.append(best_proj_t)
        tau_opt_list.append(best_tau)
    
    Zs = torch.stack(Zs_list, dim=1)  # (ns, n_pv)
    Zt = torch.stack(Zt_list, dim=1)  # (nt, n_pv)
    tau_opt = torch.tensor(tau_opt_list, device=device, dtype=Xs.dtype)

    var_s = torch.var(Zs, dim=0)
    var_t = torch.var(Zt, dim=0)
    avg_var = (var_s + var_t) / 2
    sort_idx = torch.argsort(avg_var, descending=True)

    Zs = Zs[:, sort_idx]
    Zt = Zt[:, sort_idx]
    tau_opt = tau_opt[sort_idx]
    
    # ---- 4.9 Package model ----
    model = {
        'Xs_scaler': Xs_scaler,
        'Xt_scaler': Xt_scaler,
        'Xs_scaled': Xs,  # Already on device and scaled
        'Xt_scaled': Xt,
        'rho_s': rho_s,
        'rho_t': rho_t,
        'theta': theta,
        'tau_opt': tau_opt,
        'kernel': kernel,
        'gamma': gamma,
        'cross_kernel': cross_kernel,
        'device': device,
    }
    
    return Zs, Zt, tau_opt, model

# ------------------------------------------------------------------
# 5. Transform Function (Fully GPU version)
# ------------------------------------------------------------------
def transact_transform_gpu(X_new, model, space):
    """
    Projects new data points into consensus space on GPU.
    
    Parameters
    ----------
    X_new : torch.Tensor or np.ndarray
        New data points, shape (n_new, p)
    model : dict
        Model from transact_align_gpu
    space : str
        'source' or 'target'
        
    Returns
    -------
    Z_new : torch.Tensor
        Consensus scores, shape (n_new, n_pv)
    """
    device = model['device']
    kernel = model['kernel']
    gamma = model['gamma']
    rho_s = model['rho_s']
    rho_t = model['rho_t']
    theta = model['theta']
    tau_opt = model['tau_opt']
    Xs_scaled = model['Xs_scaled']
    Xt_scaled = model['Xt_scaled']
    cross_kernel = model['cross_kernel']
    
    # Convert to tensor if needed
    if not torch.is_tensor(X_new):
        X_new = torch.from_numpy(X_new).float()
    
    X_new = X_new.to(device)
    
    if X_new.dim() == 1:
        X_new = X_new.unsqueeze(0)
    
    # Scale on GPU
    if space == 'source':
        X_new_scaled = model['Xs_scaler'].transform(X_new)
        
        # Compute kernels
        if kernel == 'rbf':
            k_new_s = rbf_kernel_torch(X_new_scaled, Xs_scaled, gamma)
            if cross_kernel is None:
                common = min(X_new_scaled.shape[1], Xt_scaled.shape[1])
                k_new_st = rbf_kernel_torch(X_new_scaled[:, :common], 
                                           Xt_scaled[:, :common], gamma)
            else:
                k_new_st = cross_kernel(X_new_scaled, Xt_scaled)
        else:  # linear
            k_new_s = linear_kernel_torch(X_new_scaled, Xs_scaled)
            k_new_st = linear_kernel_torch(X_new_scaled, Xt_scaled) if cross_kernel is None else cross_kernel(X_new_scaled, Xt_scaled)
        
        s_new = k_new_s @ rho_s.t()
        t_new = k_new_st @ rho_t.t()
        
    elif space == 'target':
        X_new_scaled = model['Xt_scaler'].transform(X_new)
        
        # Compute kernels
        if kernel == 'rbf':
            k_new_t = rbf_kernel_torch(X_new_scaled, Xt_scaled, gamma)
            if cross_kernel is None:
                common = min(Xs_scaled.shape[1], X_new_scaled.shape[1])
                k_new_ts = rbf_kernel_torch(Xs_scaled[:, :common], 
                                           X_new_scaled[:, :common], gamma).t()
            else:
                k_new_ts = cross_kernel(Xs_scaled, X_new_scaled).t()
        else:  # linear
            k_new_t = linear_kernel_torch(X_new_scaled, Xt_scaled)
            k_new_ts = linear_kernel_torch(Xs_scaled, X_new_scaled).t() if cross_kernel is None else cross_kernel(Xs_scaled, X_new_scaled).t()
        
        s_new = k_new_ts @ rho_s.t()
        t_new = k_new_t @ rho_t.t()
    else:
        raise ValueError("space must be 'source' or 'target'")
    
    # ---- Apply geodesic interpolation ----
    Z_new_list = []
    n_pv = len(theta)
    
    for q in range(n_pv):
        tau = tau_opt[q]
        
        if torch.sin(theta[q]) < 1e-8:
            GAMMA = 1 - tau
            KSI = tau
        else:
            GAMMA = torch.sin((1 - tau) * theta[q]) / torch.sin(theta[q])
            KSI = torch.sin(tau * theta[q]) / torch.sin(theta[q])
        
        proj_new = GAMMA * s_new[:, q] + KSI * t_new[:, q]
        Z_new_list.append(proj_new)
    
    return torch.stack(Z_new_list, dim=1)