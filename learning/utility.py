import os
import random
import re
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans#,DBSCAN
# from geomloss import SamplesLoss
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
import warnings
from scipy import stats
warnings.filterwarnings('ignore', message='.*ks_2samp.*')


def determine_pca_components(X_centered,limit:int=100):
    min_samples_features = min(X_centered.shape[0], X_centered.shape[1])
    if min_samples_features > limit:
        pca = PCA()
        pca.fit(X_centered)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(explained_variance >= 0.90) + 1
    else:
        num_components = min(min_samples_features, limit)
    return num_components

def partitionAndMatch(x1:np.ndarray, x2:np.ndarray,
                      system1_samples:list, system2_samples:list,
                      device,
                      n_clusters:int=8,max_iter:int=500,
                      dim_reduction:bool=False):
    if torch.is_tensor(x1):
        x1 = x1.detach().cpu().numpy()
    if torch.is_tensor(x2):    
        x2 = x2.detach().cpu().numpy()
    
    # Center data
    X1_centered = x1 - np.mean(x1, axis=0)
    X2_centered = x2 - np.mean(x2, axis=0)

    # Dimensionally reduce data
    if dim_reduction:
        number_of_components_1 = determine_pca_components(X1_centered)
        number_of_components_2 = determine_pca_components(X2_centered)
        pca_1 = PCA(n_components=number_of_components_1)
        pca_2 = PCA(n_components=number_of_components_2)
        X1_centered = pca_1.fit_transform(X1_centered)
        X2_centered = pca_2.fit_transform(X2_centered)

    ### System 1 clustering using KMeans
    clusterer1 = KMeans(n_clusters=n_clusters,max_iter=max_iter,n_init='auto')
    labels1 = clusterer1.fit_predict(X1_centered)
    partitioned_df1 = pd.DataFrame({'system1_samples': system1_samples, 'system1_label': labels1})

    ### System 2 clustering using KMeans
    clusterer2 = KMeans(n_clusters=n_clusters,max_iter=max_iter,n_init='auto')
    labels2 = clusterer2.fit_predict(X2_centered)
    partitioned_df2 = pd.DataFrame({'system2_samples': system2_samples, 'system2_label': labels2})

    ### Match system 1 clusters with system 2 clusters using optimal transport
    unique_labels1 = np.unique(labels1[labels1 >= 0])
    unique_labels2 = np.unique(labels2[labels2 >= 0])
    sinkhornLoss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    cost_matrix = np.zeros((len(unique_labels1), len(unique_labels2)))
    
    for i, label1 in enumerate(unique_labels1):
        print(i,label1)
        inds1 = np.where(labels1 == label1)[0]
        x1 = torch.tensor(X1_centered[inds1, :], dtype=torch.float32, device=device)
        for j, label2 in enumerate(unique_labels2):
            inds2 = np.where(labels2 == label2)[0]
            x2 = torch.tensor(X2_centered[inds2, :], dtype=torch.float32, device=device)
            dist = sinkhornLoss(x1, x2)
            cost_matrix[i, j] = dist.item()
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_df = pd.DataFrame({'system1_label': unique_labels1[row_ind], 'system2_label': unique_labels2[col_ind]})
    matched_df = matched_df.merge(partitioned_df1, how='left', on='system1_label')
    matched_df = matched_df.merge(partitioned_df2, how='left', on='system2_label')
    return matched_df

def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False

def find_largest_sample_len(folder_path):
    sample_folders = [f for f in os.listdir(folder_path) if f.startswith('sample_len')]
    largest_folder = max(sample_folders, key=lambda x: int(re.search(r'\d+', x).group()))
    return largest_folder

# ------------------------------------------------------------------
# 1. Helpers
# ------------------------------------------------------------------
def _center_kernel(K):
    """Double-centres a kernel matrix (Definition Supp 2.3)."""
    n = K.shape[0]
    one = np.ones((n, n)) / n
    return K - one @ K - K @ one + one @ K @ one

def _kernel_pca(Kc, n_components, eps=1e-12):
    """
    Kernel-PCA on a centred kernel.
    Returns alpha (d × n_samples) with orthonormal rows.
    """
    # full symmetric eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(Kc)
    # keep the largest n_components
    idx     = np.argsort(eigvals)[::-1][:n_components]
    eigvals = eigvals[idx]                       # shape (d,)
    eigvecs = eigvecs[:, idx]                    # shape (n, d)
    # ----- ENFORCE 2-D COLUMN SHAPE HERE -----
    eigvals = eigvals[:, np.newaxis]             # (d, 1)  ← crucial!
    # numerical safeguard for almost-zero / negative values
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    # rows = principal directions (loadings), cols = samples
    alpha = eigvecs.T / np.sqrt(eigvals + eps)   # (d, n)
    return alpha


# ------------------------------------------------------------------
# 2. Main TRANSACT alignment routine
# ------------------------------------------------------------------
def transact_align(
        Xs, Xt,
        n_src_pcs=75, n_tgt_pcs=75, n_pv=30,
        kernel='rbf', gamma=5e-4,
        cross_kernel=None,              # optional callable (Xs, Xt) → Kst
        grid=np.linspace(0., 1., 21)    # τ-grid for KS search
    ):
    """
    TRANSACT alignment: returns consensus scores (Zs, Zt) and τ*.
    Only geometry - no labels or prediction steps included.

    Addapted and implemented from: https://doi.org/10.1073/pnas.2106682118
    Mourragui, Soufiane MC, et al. 
    "Predicting patient response with models trained on cell lines and patient-derived xenografts by nonlinear transfer learning." 
    Proceedings of the National Academy of Sciences 118.49 (2021): e2106682118.
    ----------------------------------------------------------------
    Parameters
    ----------
    Xs : (ns, ps) array
    Xt : (nt, pt) array        Molecular spaces (features may differ)
    """
    # ---- 2.1  z-score inside each domain (common in the paper) ----
    Xs_scaler = StandardScaler().fit(Xs)
    Xt_scaler = StandardScaler().fit(Xt)
    Xs = Xs_scaler.transform(Xs)
    Xt = Xt_scaler.transform(Xt)

    # ---- 2.2  build within- and cross-domain kernels --------------
    if kernel == 'linear':
        Ks  = linear_kernel(Xs, Xs)
        Kt  = linear_kernel(Xt, Xt)
        Kst = linear_kernel(Xs, Xt) if cross_kernel is None else cross_kernel(Xs, Xt)
    elif kernel == 'rbf':
        Ks  = rbf_kernel(Xs, Xs, gamma=gamma)
        Kt  = rbf_kernel(Xt, Xt, gamma=gamma)
        if cross_kernel is None:
            # If feature sets differ, give user the chance to supply custom Kst;
            # otherwise fall back to intersection of shared columns.
            common = min(Xs.shape[1], Xt.shape[1])
            Kst = rbf_kernel(Xs[:, :common], Xt[:, :common], gamma=gamma)
        else:
            Kst = cross_kernel(Xs, Xt)
    else:
        raise ValueError("kernel must be 'linear' or 'rbf'")

    # ---- 2.3  centre all kernels (Proposition Supp 2.4: https://doi.org/10.1073/pnas.2106682118) ----------
    Ks_c   = _center_kernel(Ks)
    Kt_c   = _center_kernel(Kt)
    Cns    = np.eye(len(Ks)) - 1/len(Ks)
    Cnt    = np.eye(len(Kt)) - 1/len(Kt)
    Kst_c  = Cns @ Kst @ Cnt                                 # centred cross-kernel

    # ---- 2.4  Kernel-PCA → Non-linear PCs (alpha_s, alpha_t) ---------------
    alpha_s = _kernel_pca(Ks_c, n_src_pcs)                         # ℝ^{ds × ns}
    alpha_t = _kernel_pca(Kt_c, n_tgt_pcs)                         # ℝ^{dt × nt}

    # ---- 2.5  Cosine similarity matrix (Eq. [1]) ----------------
    MK = alpha_s @ Kst_c @ alpha_t.T                                    # ℝ^{ds × dt}:https://doi.org/10.1073/pnas.2106682118

    # ---- 2.6  Principal Vectors by SVD (Theorem Supp 5.3: https://doi.org/10.1073/pnas.2106682118) -------
    U, Σ, Vt = np.linalg.svd(MK, full_matrices=False)
    βs, βt = U[:, :n_pv], Vt.T[:, :n_pv]
    rho_s     = βs.T @ alpha_s                                        # sample-importance loadings rho_s, rho_t: https://doi.org/10.1073/pnas.2106682118
    rho_t     = βt.T @ alpha_t

    theta      = np.arccos(np.clip(Σ[:n_pv], 0, 1))               # principal angles θ_q (Supp 5.6)

    # ---- 2.7  Evaluate PVs on samples ----------
    Ss  = Ks_c  @ rho_s.T                                        # ℝ^{ns × d}
    Ts  = Kst_c @ rho_t.T                                        # source eval. of target PVs
    St  = Kst_c.T @ rho_s.T                                      # target eval. of source PVs
    Tt  = Kt_c  @ rho_t.T                                        # ℝ^{nt × d}

    # ---- 2.8  Per-pair geodesic interpolation & KS search -------
    ns, nt = Xs.shape[0], Xt.shape[0]
    Zs, Zt, tau_opt = [], [], []

    for q in range(n_pv):
        s_q, t_q = Ss[:, q], Tt[:, q]                         # projections on PV_q
        best_tau, best_stat = 0.5, np.inf

        for tau in grid:
            GAMMA = np.sin((1-tau)*theta[q]) / np.sin(theta[q])
            KSI = np.sin(tau*theta[q])      / np.sin(theta[q])            # Def. Supp 6.1: https://doi.org/10.1073/pnas.2106682118
            proj_s = GAMMA*s_q + KSI*St[:, q]
            proj_t = GAMMA*Ts[:, q] + KSI*t_q
            stat   = ks_2samp(proj_s, proj_t).statistic       # KS statistic as in main text: https://doi.org/10.1073/pnas.2106682118
            if stat < best_stat:
                best_stat, best_tau = stat, tau
                best_proj_s, best_proj_t = proj_s, proj_t

        Zs.append(best_proj_s)
        Zt.append(best_proj_t)
        tau_opt.append(best_tau)

    Zs = np.column_stack(Zs)                                  # consensus scores
    Zt = np.column_stack(Zt)

    # ---- 1.9  Package the model for transformation ----
    model = {
        'Xs_scaler': Xs_scaler,
        'Xt_scaler': Xt_scaler,
        'Xs_scaled': Xs, # Store scaled reference data
        'Xt_scaled': Xt,
        'rho_s': rho_s,
        'rho_t': rho_t,
        'theta': theta,
        'tau_opt': np.array(tau_opt),
        'kernel': kernel,
        'gamma': gamma,
        'cross_kernel': cross_kernel,
    }

    return Zs, Zt, np.array(tau_opt), model

# ------------------------------------------------------------------
# 3. Transform Function
# ------------------------------------------------------------------
def transact_transform(X_new, model, space):
    """
    Projects new data points into the consensus space defined by a fitted TRANSACT model ( https://doi.org/10.1073/pnas.2106682118 ).

    Parameters
    ----------
    X_new : np.ndarray
        New data points to project, shape (n_new, p).
    model : dict
        The model object returned by transact_fit.
    space : str, 'source' or 'target'
        Specifies whether the new data belongs to the source or target space.

    Returns
    -------
    Z_new : np.ndarray
        The consensus scores for the new data points, shape (n_new, n_pv).
    """
    # ---- 3.1  Unpack model ----
    kernel = model['kernel']
    gamma = model['gamma']
    rho_s, rho_t = model['rho_s'], model['rho_t']
    theta, tau_opt = model['theta'], model['tau_opt']
    Xs_scaled, Xt_scaled = model['Xs_scaled'], model['Xt_scaled']
    cross_kernel = model['cross_kernel']

    # Ensure X_new is 2D
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    # ---- 3.2  Scale new data and compute kernels ----
    if space == 'source':
        X_new_scaled = model['Xs_scaler'].transform(X_new)
        # Kernel of new points against original reference sets
        k_new_s = rbf_kernel(X_new_scaled, Xs_scaled, gamma=gamma) if kernel == 'rbf' else linear_kernel(X_new_scaled, Xs_scaled)
        if cross_kernel is None:
            common = min(X_new_scaled.shape[1], Xt_scaled.shape[1])
            k_new_st = rbf_kernel(X_new_scaled[:,:common], Xt_scaled[:,:common], gamma=gamma) if kernel == 'rbf' else linear_kernel(X_new_scaled, Xt_scaled)
        else:
            k_new_st = cross_kernel(X_new_scaled, Xt_scaled)

        # ---- 3.3  Project onto PVs using the kernel trick ----
        # This is the out-of-sample extension: K(x_new, x_train) @ coeffs
        s_new = k_new_s @ rho_s.T
        t_new = k_new_st @ rho_t.T

    elif space == 'target':
        X_new_scaled = model['Xt_scaler'].transform(X_new)
        # Kernel of new points against original reference sets
        k_new_t = rbf_kernel(X_new_scaled, Xt_scaled, gamma=gamma) if kernel == 'rbf' else linear_kernel(X_new_scaled, Xt_scaled)
        if cross_kernel is None:
             common = min(Xs_scaled.shape[1], X_new_scaled.shape[1])
             k_new_ts = rbf_kernel(Xs_scaled[:,:common], X_new_scaled[:,:common], gamma=gamma).T if kernel == 'rbf' else linear_kernel(Xs_scaled, X_new_scaled).T
        else:
             k_new_ts = cross_kernel(Xs_scaled, X_new_scaled).T # Transpose for correct shape

        # ---- 3.3  Project onto PVs using the kernel trick ----
        s_new = k_new_ts @ rho_s.T
        t_new = k_new_t @ rho_t.T
    else:
        raise ValueError("space must be 'source' or 'target'")

    # ---- 3.4  Apply saved geodesic interpolation ----
    Z_new_list = []
    for q in range(len(theta)):
        tau = tau_opt[q]
        if np.sin(theta[q]) == 0:
            GAMMA, KSI = (1 - tau), tau
        else:
            GAMMA = np.sin((1 - tau) * theta[q]) / np.sin(theta[q])
            KSI = np.sin(tau * theta[q]) / np.sin(theta[q])

        # Interpolate the two projections for the new point
        proj_new = GAMMA * s_new[:, q] + KSI * t_new[:, q]
        Z_new_list.append(proj_new)

    return np.column_stack(Z_new_list)
