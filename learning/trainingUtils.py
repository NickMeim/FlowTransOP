import numpy as np
import torch
from evaluationUtils import pearson_r, r_square
from utility import *
from transact_utility_gpu import *
from sklearn.metrics import confusion_matrix, r2_score
from scipy.stats import pearsonr
from torch.distributions import NegativeBinomial
from sklearn.metrics import confusion_matrix, r2_score
from scipy.stats import pearsonr
from torch.distributions import NegativeBinomial
# from geomloss import SamplesLoss
import logging
logger = logging.getLogger(__name__)  # Use __name__ for module-level logger
logger.setLevel(logging.INFO)
print2log = logger.info

def rbf_kernel(X: torch.Tensor,sigma: float | None = None) -> torch.Tensor:
    """
    Compute the RBF (Gaussian) kernel matrix K for X:
      K_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)).
    If sigma is None, use the median heuristic:
      sigma = sqrt(0.5 * median(pairwise_sq_dists)).
    """
    # pairwise squared Euclidean distances (n×n)
    D2 = torch.cdist(X, X, p=2).pow(2)
    if sigma is None:
        # median of off-diagonal distances
        # add a small epsilon to avoid zero
        med = D2[~torch.eye(D2.size(0), dtype=bool, device=X.device)].median()
        sigma = torch.sqrt(0.5 * med)
    gamma = 1.0 / (2 * sigma * sigma)
    return torch.exp(-gamma * D2)

def hsic(X: torch.Tensor,
         Y: torch.Tensor,
         sigma_x: float | None = None,
         sigma_y: float | None = None) -> torch.Tensor:
    """
    Biased HSIC estimator between X and Y.
    HSIC = (1 / (n-1)^2) * trace( Kc @ Lc )
    where Kc = H K H, Lc = H L H, H = I - (1/n) 11^T.
    X : (n, m1) data matrix
    Y : (n, m2) data matrix
    sigma_x, sigma_y: RBF kernel widths (None → median heuristic)
    Returns
    ------
    scalar HSIC value
    """
    n = X.size(0)
    # 1) Gram matrices
    K = rbf_kernel(X, sigma_x)
    L = rbf_kernel(Y, sigma_y)
    # 2) centering matrix H = I - 1/n 11^T
    I = torch.eye(n, device=X.device, dtype=X.dtype)
    ones = torch.ones((n, n), device=X.device, dtype=X.dtype) / n
    H = I - ones
    # 3) centered kernels
    Kc = H @ K @ H
    Lc = H @ L @ H
    # 4) HSIC
    return torch.trace(Kc @ Lc) /torch.sqrt(torch.trace(Kc @ Kc) * torch.trace(Lc @ Lc)) #torch.trace(Kc @ Lc) / (n - 1)**2

def cross_view_similarity(X1, X2, k=None, center=True):
    """
    X1 : (n, d1) tensor   – view 1
    X2 : (n, d2) tensor   – view 2   (same n rows)
    k  : int|None         – latent dimensionality (None → full rank)
    Returns
    -------
    C_tilde : (n, n)   sample × sample similarity matrix
    Z1, Z2  : (n, k)   latent representations of each view
    """
    if center:
        X1 = X1 - X1.mean(0, keepdim=True)
        X2 = X2 - X2.mean(0, keepdim=True)
    # cross-covariance SVD  (fast on GPU, batched-SVD aware)
    Sigma12 = X1.T @ X2 / (X1.size(0) - 1)
    U, _, Vh = torch.linalg.svd(Sigma12, full_matrices=False)
    V = Vh.T
    if k is not None:
        U, V = U[:, :k], V[:, :k]
    Z1 = X1 @ U           # n × k
    Z2 = X2 @ V           # n × k
    C_tilde = Z1 @ Z2.T   # n × n
    return C_tilde #, Z1, Z2

def compute_C_tilde(xcentered_1, xcentered_2, d=1000):
    """
    Compute approximate similarity matrix for datasets with different features
    using Random Projection.
    Args:
        xcentered_1: Centered tensor of shape (n1, p)
        xcentered_2: Centered tensor of shape (n2, q)
        d: Projection dimension (default=1000) 
    Returns:
        C_tilde: Approximate similarity matrix of shape (n1, n2)
    """
    p = xcentered_1.size(1)
    q = xcentered_2.size(1)
    # Generate random projection matrices (fixed during training)
    W1 = torch.randn(p, d, device=xcentered_1.device, dtype=xcentered_1.dtype) / (d**0.5)
    W2 = torch.randn(q, d, device=xcentered_2.device, dtype=xcentered_2.dtype) / (d**0.5)
    # Project to common d-dimensional space
    X1_proj = xcentered_1 @ W1
    X2_proj = xcentered_2 @ W2
    # Compute approximate cross-covariance
    C_tilde = X1_proj @ X2_proj.t()
    return C_tilde



# ---------- power iteration ----------
def power_iteration(C: torch.Tensor,
                    num_iters: int = 10,
                    eps: float = 1e-6) -> torch.Tensor:
    """
    Returns an approximation of the dominant eigen-vector of C.
    Args
    ----
    C : [d_z, d_z]  symmetric (or at least square) matrix.
    num_iters : how many iterations of power method to run
    eps : tiny value to avoid division by zero if C is (near) rank-deficient
    """
    # 1. random starting vector, same dtype / device as C
    b_k = torch.randn(C.shape[0], device=C.device, dtype=C.dtype)
    b_k /= (b_k.norm() + eps)          # normalise once
    # 2. power iterations
    for _ in range(num_iters):
        b_k1 = C @ b_k                 # multiply by the matrix
        b_k1_norm = b_k1.norm()
        if b_k1_norm < eps:
            break                      # C is (almost) zero in this direction
        b_k = b_k1 / b_k1_norm         # renormalise
    return b_k     

### Taken from https://arxiv.org/abs/2506.16895 on November 7th, 2025
### Reference: Gröger, Fabian, Shuo Wen, Huyen Le, and Maria Brbić. "With Limited Data for Multimodal Alignment, Let the STRUCTURE Guide You." arXiv preprint arXiv:2506.16895 (2025).
def reg_structure(X, A, L=1, tau=.05, eps=1e-8):
    X_hat = torch.nn.functional.normalize(X, p=2, dim=1, eps=eps)
    A_hat = torch.nn.functional.normalize(A, p=2, dim=1, eps=eps)
    X_tilde = X_hat - X_hat.mean(dim=0, keepdim=True)
    A_tilde = A_hat - A_hat.mean(dim=0, keepdim=True)
    Sx = (X_tilde @ X_tilde.T) / tau
    Sa = (A_tilde @ A_tilde.T) / tau
    Px = torch.nn.functional.softmax(Sx, dim=1)
    Pa = torch.nn.functional.softmax(Sa, dim=1)
    r_S = 0.0
    for l in range(1, L + 1):
        Px_l, Pa_l = Px.matrix_power(l), Pa.matrix_power(l)
        M_l = 0.5 * (Px_l + Pa_l)
        d_js = 0.5 * ((Pa_l * (torch.log(Pa_l + eps) - torch.log(M_l + eps))).sum()
        + (Px_l * (torch.log(Px_l + eps) - torch.log(M_l + eps))).sum())
        r_S += d_js / l
    return r_S / L

def _l2_safe(module, reg):
    """Call .L2Regularization on the module if it has one, else sum over children that do."""
    if hasattr(module, 'L2Regularization') and not isinstance(module, torch.nn.Sequential):
        return module.L2Regularization(reg)
    total = 0.0
    for m in module.modules():
        if m is module:
            continue
        if hasattr(m, 'L2Regularization'):
            total = total + m.L2Regularization(reg)
    return total

class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class MultipleScheduler:
    def __init__(self, *op):
        self.optimizers = op

    def step(self):
        for op in self.optimizers:
            op.step()


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)

    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

    return mmd  #

# Create a train generators
def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    while len(order) > 0:
        outList.append(order[:batchSize])
        order = order[batchSize:]
    if len(outList[-1]) < batchSize:
        outList[-1] = np.append(outList[-1], np.random.permutation(N)[0:(batchSize - len(outList[-1]))])
    return outList

def _fix_batch_size(idx, target_bs, N):
    """Ensure batch has exactly target_bs indices by padding (with replacement) or trimming."""
    idx = np.asarray(idx)
    if idx.size < target_bs:
        need = target_bs - idx.size
        extra = np.random.choice(N, size=need, replace=True)
        idx = np.concatenate([idx, extra])
    elif idx.size > target_bs:
        idx = np.random.choice(idx, size=target_bs, replace=False)
    return idx

def compute_gradients(output, input):
    grads = torch.autograd.grad(output, input, create_graph=True)
    grads = grads[0].pow(2).mean()
    return grads

def freeze_model(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad

def contrastive_loss(x, y, margin=1.0):
    pairwise_dist = torch.mean(torch.cdist(x, y))
    return pairwise_dist

# Taken and implenented from https://github.com/facebookresearch/CPA
class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, mu, y, theta, eps=1e-8):
        """Negative binomial negative log-likelihood. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3).
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        log_theta_mu_eps = torch.log(theta + mu + eps)
        res = (
                theta * (torch.log(theta + eps) - log_theta_mu_eps)
                + y * (torch.log(mu + eps) - log_theta_mu_eps)
                + torch.lgamma(y + theta)
                - torch.lgamma(theta)
                - torch.lgamma(y + 1)
        )
        res = _nan2inf(res)
        return -torch.mean(res)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

# Taken and implenented from https://github.com/facebookresearch/CPA
def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    r"""NB parameterizations conversion
    Parameters
    ----------
    mu :
        mean of the NB distribution.
    theta :
        inverse overdispersion.
    eps :
        constant used for numerical log stability. (Default value = 1e-6)
    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
    assert (mu is None) == (
        theta is None
    ), "If using the mu/theta NB parameterization, both parameters must be specified"
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits

def validate_fold(device, x_1_test, x_2_test,
                              decoder_1, decoder_2, encoder_1, encoder_2, classifier, Vsp,pairs_val=None): #encoder_2, classifier, Vsp,pairs_val=None
    # Evaluation mode
    encoder_1.eval()
    encoder_2.eval()
    decoder_1.eval()
    decoder_2.eval()
    classifier.eval()
    Vsp.eval()
    
    with torch.no_grad():
        # Prepare test data
        x_1 = x_1_test.double().to(device)
        x_2 = x_2_test.double().to(device)
        
        z_species_1 = torch.cat((torch.ones(x_1.shape[0], 1), torch.zeros(x_1.shape[0], 1)), 1).double().to(device)
        z_species_2 = torch.cat((torch.zeros(x_2.shape[0], 1), torch.ones(x_2.shape[0], 1)), 1).double().to(device)
        
        # Generate latent variables
        z_latent_base_1 = encoder_1(x_1)
        z_latent_base_2 = encoder_2(x_2)
        z_latent_1 = Vsp(z_latent_base_1, z_species_1)
        z_latent_2 = Vsp(z_latent_base_2, z_species_2)

        # reconstruction results
        y_pred_1 = decoder_1(z_latent_1)
        y_pred_2 = decoder_2(z_latent_2)
        # y_pred_1 = decoder_1(z_latent_base_1)
        # y_pred_2 = decoder_2(z_latent_base_2)
        # evaluate pearson correlation (pearson_r) of reconstruction
        r_1 = pearson_r(y_pred_1.flatten(), x_1.flatten()).detach().cpu().numpy()
        r_2 = pearson_r(y_pred_2.flatten(), x_2.flatten()).detach().cpu().numpy()

        # Classification results
        labels = classifier(torch.cat((z_latent_1, z_latent_2), 0))
        true_labels = torch.cat((torch.ones(z_latent_1.shape[0]).view(z_latent_1.shape[0], 1),
                               torch.zeros(z_latent_2.shape[0]).view(z_latent_2.shape[0], 1)), 0).long()
        
        _, predicted = torch.max(labels, 1)
        predicted = predicted.cpu().numpy()
        cf_matrix = confusion_matrix(true_labels.numpy(), predicted)
        tn, fp, fn, tp = cf_matrix.ravel()
        class_acc = (tp + tn) / predicted.size
        f1 = 2 * tp / (2 * tp + fp + fn)

        # translation results
        if pairs_val is not None:
            x_1_equivalent = x_1[pairs_val,:]
            x_2_equivalent = x_2[pairs_val,:]
            z_species_1_equivalent = z_species_1[pairs_val,:]
            z_species_2_equivalent = z_species_2[pairs_val,:]
            # first translate system 1 to 2
            z_latent_base_1_equivalent  = encoder_1(x_1_equivalent)
            z_latent_1_equivalent = Vsp(z_latent_base_1_equivalent,1.-z_species_1_equivalent)
            x_hat_2_equivalent = decoder_2(z_latent_1_equivalent).detach()
            # x_hat_2_equivalent = decoder_2(z_latent_base_1_equivalent).detach()
            pearson_1_to_2 = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()
            # second translate system 2 to 1
            z_latent_base_2_equivalent  = encoder_2(x_2_equivalent)
            z_latent_2_equivalent = Vsp(z_latent_base_2_equivalent,1.-z_species_2_equivalent)
            x_hat_1_equivalent = decoder_1(z_latent_2_equivalent).detach()
            # x_hat_1_equivalent = decoder_1(z_latent_base_2_equivalent).detach()
            pearson_2_to_1 = pearson_r(x_hat_1_equivalent.flatten(), x_1_equivalent.flatten()).detach().cpu().numpy()
        else:
            pearson_1_to_2 = np.nan
            pearson_2_to_1 = np.nan
    
    return pearson_1_to_2, pearson_2_to_1,f1, class_acc, r_1, r_2 #pearson_2_to_1,f1, class_acc

def train_fold(model_params, device, x_1_train, x_2_train,
                    decoder_1, decoder_2, encoder_1, encoder_2, adverse_classifier,classifier, Vsp,
                    bs_1:int, bs_2:int, NUM_EPOCHS:int,
                    class_criterion,
                    pairs_train=None):    
    # Combine parameters and create optimizers/schedulers
    allParams = list(decoder_1.parameters()) + list(decoder_2.parameters()) + \
                list(encoder_1.parameters()) + list(encoder_2.parameters()) + \
                list(classifier.parameters()) + list(Vsp.parameters())
    
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    optimizer_adv = torch.optim.Adam(adverse_classifier.parameters(), lr=model_params['adv_lr'], weight_decay=0)
    
    if model_params['schedule_step_adv'] is not None:
        scheduler_adv = torch.optim.lr_scheduler.StepLR(optimizer_adv,
                                                        step_size=model_params['schedule_step_adv'],
                                                        gamma=model_params['gamma_adv'])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])

    # Get dataset sizes
    N_1 = x_1_train.shape[0]
    N_2 = x_2_train.shape[0]

    ## distance criterion
    sampleLoss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    
    # Training loop
    for e in range(NUM_EPOCHS):        
        decoder_1.train()
        decoder_2.train()
        encoder_1.train()
        encoder_2.train()
        classifier.train()
        adverse_classifier.train()
        Vsp.train()

        # Generate training batches
        trainloader_1 = getSamples(N_1, bs_1)
        trainloader_2 = getSamples(N_2, bs_2)
        maxLen = max(len(trainloader_1), len(trainloader_2))
        
        # Pad shorter trainloader if needed
        if len(trainloader_1) < maxLen:
            while len(trainloader_1) < maxLen:
                trainloader_1 += getSamples(N_1, bs_1)[:maxLen - len(trainloader_1)]
        if len(trainloader_2) < maxLen:
            while len(trainloader_2) < maxLen:
                trainloader_2 += getSamples(N_2, bs_2)[:maxLen - len(trainloader_2)]
        
        # Iterate through batches
        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]
            
            X_1 = x_1_train[dataIndex_1, :].double().to(device)
            X_2 = x_2_train[dataIndex_2, :].double().to(device)

            
            # Species vectors
            z_species_1 = torch.cat((torch.ones(X_1.shape[0], 1), torch.zeros(X_1.shape[0], 1)), 1).double().to(device)
            z_species_2 = torch.cat((torch.zeros(X_2.shape[0], 1), torch.ones(X_2.shape[0], 1)), 1).double().to(device)
            
            optimizer.zero_grad()
            optimizer_adv.zero_grad()
                        
            if e % model_params['schedule_step_adv'] == 0:
                # for _ in range(20):
                z_base_1 = encoder_1(X_1)
                z_base_2 = encoder_2(X_2)
                latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
                labels_adv = adverse_classifier(latent_base_vectors)
                true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                                         torch.zeros(z_base_2.shape[0])),0).long().to(device)
                _, predicted = torch.max(labels_adv, 1)
                predicted = predicted.cpu().numpy()
                cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
                tn, fp, fn, tp = cf_matrix.ravel()
                f1_basal_trained = 2*tp/(2*tp+fp+fn)
                adv_entropy = class_criterion(labels_adv,true_labels)
                adversary_drugs_penalty = compute_gradients(labels_adv.sum(), latent_base_vectors)
                loss_adv = adv_entropy + model_params['adv_penalnty'] * adversary_drugs_penalty
                loss_adv.backward()
                optimizer_adv.step()
            else:
                #optimizer_adv.zero_grad()
                #for _ in range(5):
                #    z_base_1 = encoder_1(X_1)
                #    z_base_2 = encoder_2(X_2)
                #    latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
                #    labels_adv = adverse_classifier(latent_base_vectors)
                #    true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                #        torch.zeros(z_base_2.shape[0])),0).long().to(device)
                #    _, predicted = torch.max(labels_adv, 1)
                #    predicted = predicted.cpu().numpy()
                #    cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
                #    tn, fp, fn, tp = cf_matrix.ravel()
                #    f1_basal_trained = 2*tp/(2*tp+fp+fn)
                #    adv_entropy = class_criterion(labels_adv,true_labels)
                #    adversary_drugs_penalty = compute_gradients(labels_adv.sum(), latent_base_vectors)
                #    loss_adv = adv_entropy + model_params['adv_penalnty'] * adversary_drugs_penalty
                #    loss_adv.backward()
                #    optimizer_adv.step()
                # now perform the non-aversesary step    
                optimizer.zero_grad()
                z_base_1 = encoder_1(X_1)
                z_base_2 = encoder_2(X_2)
                latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
                
                z_1 = Vsp(z_base_1,z_species_1)
                z_2 = Vsp(z_base_2,z_species_2)
                latent_vectors = torch.cat((z_1, z_2), 0)
                
                y_pred_1 = decoder_1(z_1)
                # y_pred_1 = decoder_1(z_base_1)
                fitLoss_1 = torch.mean(torch.sum((y_pred_1 - X_1)**2,dim=1))
                L2Loss_1 = decoder_1.L2Regularization(model_params['dec_l2_reg']) + encoder_1.L2Regularization(model_params['enc_l2_reg'])
                loss_1 = fitLoss_1 + L2Loss_1
                
                y_pred_2 = decoder_2(z_2)
                # y_pred_2 = decoder_2(z_base_2)
                fitLoss_2 = torch.mean(torch.sum((y_pred_2 - X_2)**2,dim=1))
                L2Loss_2 = decoder_2.L2Regularization(model_params['dec_l2_reg']) + encoder_2.L2Regularization(model_params['enc_l2_reg'])
                loss_2 = fitLoss_2 + L2Loss_2

                # Classification loss
                labels = classifier(latent_vectors)
                true_labels = torch.cat((torch.ones(z_1.shape[0]),
                   torch.zeros(z_2.shape[0])),0).long().to(device)
                entropy = class_criterion(labels,true_labels)
                _, predicted = torch.max(labels, 1)
                predicted = predicted.cpu().numpy()
                cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
                tn, fp, fn, tp = cf_matrix.ravel()
                f1_latent = 2*tp/(2*tp+fp+fn)
                
                # Remove signal from z_basal
                labels_adv = adverse_classifier(latent_base_vectors)
                true_labels = torch.cat((torch.ones(z_base_1.shape[0]),
                    torch.zeros(z_base_2.shape[0])),0).long().to(device)
                adv_entropy = class_criterion(labels_adv,true_labels)
                _, predicted = torch.max(labels_adv, 1)
                predicted = predicted.cpu().numpy()
                cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
                tn, fp, fn, tp = cf_matrix.ravel()
                f1_basal = 2*tp/(2*tp+fp+fn)

                ## minimize Wasserstein distance
                #+ model_params['distance_reg']*dist_loss
                dist_loss = sampleLoss(z_base_1,z_base_2) # approximation of wasserstein distance
                dist_loss = dist_loss.sum()
                # #dist_loss = contrastive_loss(z_base_1,z_base_2)

                # Domain adaptation loss
                _,S1,V1 = torch.svd(z_base_1)
                _,S2,V2 = torch.svd(z_base_2)
                if (S1.shape[0] < S2.shape[0]):
                    S2 = S2[:S1.shape[0]]
                    V2 = V2[:,:S1.shape[0]]
                if (S2.shape[0] < S1.shape[0]):
                    S1 = S1[:S2.shape[0]]
                    V1 = V1[:,:S2.shape[0]]
                singularValLoss = torch.sum(torch.square(S1-S2))
                singularVecLoss = torch.sum(torch.square(V1-V2))
                # A = torch.clone(z_base_1)
                # B = torch.clone(z_base_2)
                # if X_1.shape[0] < X_2.shape[0]:
                #     padding = torch.zeros((X_2.shape[0] - X_1.shape[0], z_base_1.shape[1])).to(device)
                #     A = torch.cat((z_base_1, padding), dim=0)
                # if X_2.shape[0] < X_1.shape[0]:
                #     padding = torch.zeros((X_1.shape[0] - X_2.shape[0], z_base_2.shape[1])).to(device)
                #     B = torch.cat((z_base_2, padding), dim=0)
                # M1 = A.t() @ B
                # M2 = B.t() @ A
                # S1 = z_base_1.t() @ z_base_1
                # S2 = z_base_2.t() @ z_base_2
                # adaptation_loss = torch.sum(torch.square(M1-torch.eye(M1.shape[0]).to(device))) + torch.sum(torch.square(M2-torch.eye(M2.shape[0]).to(device)))
                # orthogonality_loss = torch.sum(torch.square(S1-torch.eye(S1.shape[0]).to(device))) + torch.sum(torch.square(S2-torch.eye(S2.shape[0]).to(device)))
                
                loss = loss_1 + loss_2  + \
                        model_params['distance_reg']*dist_loss - model_params['reg_adv']*adv_entropy  + \
                        model_params['reg_classifier'] * entropy+classifier.L2Regularization(model_params['state_class_reg'])  + \
                        Vsp.Regularization(model_params['v_reg']) + \
                        model_params['reg_adapt']*(singularValLoss+singularVecLoss)
                # loss = loss_1 + loss_2  + \
                #         model_params['reg_classifier'] * entropy- model_params['reg_adv']*adv_entropy + classifier.L2Regularization(model_params['state_class_reg'])  + \
                #         Vsp.Regularization(model_params['v_reg']) + \
                #         model_params['reg_adapt']*(singularValLoss+singularVecLoss)
                loss.backward()
                optimizer.step()

                pearson_1 = torch.nanmean(pearson_r(y_pred_1.detach(), X_1.detach()))
                r2_1 = r_square(y_pred_1.detach().flatten(), X_1.detach().flatten())
                mse_1 = torch.mean(torch.mean((y_pred_1.detach() - X_1.detach())**2,dim=1))
            
                pearson_2 = torch.nanmean(pearson_r(y_pred_2.detach(), X_2.detach()))
                r2_2 = r_square(y_pred_2.detach().flatten(), X_2.detach().flatten())
                mse_2 = torch.mean(torch.mean((y_pred_2.detach() - X_2.detach())**2,dim=1))

        # Adjust learning rate if needed
        if model_params['schedule_step_adv'] is not None:
            scheduler_adv.step()
        if (e>0):
            scheduler.step()
            outString = 'Epoch={:.0f}/{:.0f}'.format(e+1,NUM_EPOCHS)
            outString += ', r2_1={:.4f}'.format(r2_1.item())
            outString += ', pearson_1={:.4f}'.format(pearson_1.item())
            outString += ', MSE_1={:.4f}'.format(mse_1.item())
            outString += ', r2_2={:.4f}'.format(r2_2.item())
            outString += ', pearson_2={:.4f}'.format(pearson_2.item())
            outString += ', MSE_2={:.4f}'.format(mse_2.item())
            outString += ', Entropy Loss={:.4f}'.format(entropy.item())
            outString += ', Adverse Entropy={:.4f}'.format(adv_entropy.item())
            outString += ', loss={:.4f}'.format(loss.item())
            outString += ', F1 latent={:.4f}'.format(f1_latent)
            outString += ', F1 basal={:.4f}'.format(f1_basal)
            outString += ', F1 basal trained={:.4f}'.format(f1_basal_trained)
            outString += ',singular value loss={:.4f}'.format(singularValLoss.item())
            outString += ',singular vector loss={:.4f}'.format(singularVecLoss.item())
            outString += ',distance loss={:.4f}'.format(dist_loss.item())

        # Logging
        if (e % 250 == 0 and e > 0) or (e == 1) or (e + 1 == NUM_EPOCHS):
            print2log(outString)
        
    # evaluate
    encoder_1.eval()
    encoder_2.eval()
    decoder_1.eval()
    decoder_2.eval()
    classifier.eval()
    Vsp.eval()
    with torch.no_grad():
        z_base_1 = encoder_1(x_1_train.double().to(device))
        z_base_2 = encoder_2(x_2_train.double().to(device))
        z_species_1 = torch.cat((torch.ones(x_1_train.shape[0], 1), torch.zeros(x_1_train.shape[0], 1)), 1).double().to(device)
        z_species_2 = torch.cat((torch.zeros(x_2_train.shape[0], 1), torch.ones(x_2_train.shape[0], 1)), 1).double().to(device)
        z_1 = Vsp(z_base_1,z_species_1)
        z_2 = Vsp(z_base_2,z_species_2)
        y_pred_1 = decoder_1(z_1)
        y_pred_2 = decoder_2(z_2)
        # y_pred_1 = decoder_1(z_base_1)
        # y_pred_2 = decoder_2(z_base_2)
        pear_1 = pearson_r(y_pred_1.flatten().detach(), x_1_train.flatten().double().to(device)).cpu().numpy()
        pear_2 = pearson_r(y_pred_2.flatten().detach(), x_2_train.flatten().double().to(device)).cpu().numpy()
        # Classification results
        labels = classifier(torch.cat((z_1, z_2), 0))
        true_labels = torch.cat((torch.ones(z_1.shape[0]).view(z_1.shape[0], 1),
           torch.zeros(z_2.shape[0]).view(z_2.shape[0], 1)), 0).long()
        _, predicted = torch.max(labels, 1)
        predicted = predicted.cpu().numpy()
        cf_matrix = confusion_matrix(true_labels.numpy(), predicted)
        tn, fp, fn, tp = cf_matrix.ravel()
        class_acc = (tp + tn) / predicted.size
        f1 = 2 * tp / (2 * tp + fp + fn)

        # translation results
        if pairs_train is not None:
            x_1_equivalent = x_1_train[pairs_train,:].double()
            x_2_equivalent = x_2_train[pairs_train,:].double()
            z_species_1_equivalent = z_species_1[pairs_train,:]
            z_species_2_equivalent = z_species_2[pairs_train,:]
            # first translate system 1 to 2
            z_latent_base_1_equivalent  = encoder_1(x_1_equivalent)
            z_latent_1_equivalent = Vsp(z_latent_base_1_equivalent,1.-z_species_1_equivalent)
            x_hat_2_equivalent = decoder_2(z_latent_1_equivalent).detach()
            # x_hat_2_equivalent = decoder_2(z_latent_base_1_equivalent).detach()
            pearson_1_to_2 = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()
            # second translate system 2 to 1
            z_latent_base_2_equivalent  = encoder_2(x_2_equivalent)
            z_latent_2_equivalent = Vsp(z_latent_base_2_equivalent,1.-z_species_2_equivalent)
            x_hat_1_equivalent = decoder_1(z_latent_2_equivalent).detach()
            # x_hat_1_equivalent = decoder_1(z_latent_base_2_equivalent).detach()
            pearson_2_to_1 = pearson_r(x_hat_1_equivalent.flatten(), x_1_equivalent.flatten()).detach().cpu().numpy()

            # cosine of paired conditions
            cosineLoss = torch.nn.functional.cosine_similarity(z_latent_base_1_equivalent,z_latent_base_2_equivalent,dim=-1).mean()
            print2log('cosine of trained paired conditions: {:.4f}'.format(cosineLoss.item()))
        else:
            pearson_1_to_2 = np.nan
            pearson_2_to_1 = np.nan

    return (pearson_1_to_2, pearson_2_to_1,pear_1,pear_2,f1,class_acc,decoder_1, decoder_2, encoder_1, encoder_2, classifier, adverse_classifier, Vsp) #pear_2,f1,class_acc and encoder_2, classifier, adverse_classifier, Vsp

def train_flowMatch_fold(model_params, device, x_1_train, x_2_train,z_1_train,z_2_train,
                    decoder_1, decoder_2,flow,
                    bs_1:int, bs_2:int, NUM_EPOCHS:int,
                    pairs_train=None,
                    tanslation_direction = '1 to 2'):
    # Combine parameters and create optimizers/schedulers
    allParams = list(flow.parameters())
    
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    flow_loss_fn = torch.nn.MSELoss()
    decoder_1.eval()
    decoder_2.eval()

    # Get dataset sizes
    N_1 = x_1_train.shape[0]
    N_2 = x_2_train.shape[0]
    all_losses = []
    all_flow_losses = []
    all_dist_losses = []
    # Training loop
    # loss_fn = torch.nn.MSELoss()
    for e in range(NUM_EPOCHS):        
        flow.train()
        # Generate training batches
        trainloader_1 = getSamples(N_1, bs_1)
        trainloader_2 = getSamples(N_2, bs_2)
        maxLen = max(len(trainloader_1), len(trainloader_2))
        # Pad shorter trainloader if needed
        if len(trainloader_1) < maxLen:
            while len(trainloader_1) < maxLen:
                trainloader_1 += getSamples(N_1, bs_1)[:maxLen - len(trainloader_1)]
        if len(trainloader_2) < maxLen:
            while len(trainloader_2) < maxLen:
                trainloader_2 += getSamples(N_2, bs_2)[:maxLen - len(trainloader_2)]
        # Iterate through batches
        batch_losses = []
        batch_flow_losses = []
        batch_dist_losses = []
        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]
            # Get data for the batch
            X_1 = x_1_train[dataIndex_1, :]#.cpu().numpy()#.double().to(device)
            X_2 = x_2_train[dataIndex_2, :]#.cpu().numpy()#.double().to(device)
            z_1 = z_1_train[dataIndex_1, :]#.double().to(device)
            z_2 = z_2_train[dataIndex_2, :]#.double().to(device)
            # X_1_aligned, X_2_aligned, tau = transact_align_gpu(
            #     X_1,        # source → will become Z_source
            #     X_2,        # target → will become Z_target
            #     n_src_pcs=75,
            #     n_tgt_pcs=75,
            #     n_pv=30,
            #     kernel='rbf',
            #     gamma=5e-4      # or whatever you tuned
            # )
            X_2_aligned, X_1_aligned, _,_ = transact_align_gpu(
                X_2,        # source → will become Z_source
                X_1,        # target → will become Z_target
                n_src_pcs=75,
                n_tgt_pcs=75,
                n_pv=30,
                kernel='rbf',
                gamma=5e-4      # or whatever you tuned
            )
            # X_1_aligned = torch.from_numpy(X_1_aligned).double().to(device)
            # X_2_aligned = torch.from_numpy(X_2_aligned).double().to(device)
            X_1_aligned = X_1_aligned  - X_1_aligned.mean(0)
            X_2_aligned = X_2_aligned  - X_2_aligned.mean(0)
            C = X_1_aligned @ X_2_aligned.T
            # Using z_1 and z_2 as the latent representations for flow matching
            # Then in each iteration:
            C = C/C.max()
            optimizer.zero_grad()
            if tanslation_direction == '1 to 2':
                # Translate z1 to z2 with flow
                if z_1.shape[0] == z_2.shape[0]:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_1 + t * z_2
                    dz_t = z_2 - z_1
                elif z_1.shape[0] > z_2.shape[0]:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_1[:z_2.shape[0]] + t * z_2
                    dz_t = z_2 - z_1[:z_2.shape[0]]
                else:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_1 + t * z_2[:z_1.shape[0]]
                    dz_t = z_2[:z_1.shape[0]] - z_1
                flow_loss = flow_loss_fn(flow(z_t, t), dz_t)
                ## Conditioned based on initial correlation
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                z12 = z_1.clone()
                for step in range(n_steps):
                    z12 = flow.step(z12, time_steps[step], time_steps[step + 1])
                dist = torch.cdist(z12, z_2) * torch.nn.functional.relu(C)
                dist = torch.sum(dist)
            else:
                # Translate now z2 to z1
                if z_1.shape[0] == z_2.shape[0]:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_2 + t * z_1
                    dz_t = z_1 - z_2
                elif z_2.shape[0] > z_1.shape[0]:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_2[:z_1.shape[0]] + t * z_1
                    dz_t = z_1 - z_2[:z_1.shape[0]]
                else:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_2 + t * z_1[:z_2.shape[0]]
                    dz_t = z_1[:z_2.shape[0]] - z_2
                flow_loss = flow_loss_fn(flow(z_t, t), dz_t)
                ## Conditioned based on initial correlation
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                z21 = z_2.clone()
                for step in range(n_steps):
                    z21 = flow.step(z21, time_steps[step], time_steps[step + 1])
                dist = torch.cdist(z_1, z21) * torch.nn.functional.relu(C)
                dist = torch.sum(dist)
            # # Sample from the prior distribution in the *target* space (R^m2)
            # z_0 = torch.randn_like(z_2, device=device, dtype=flow.dtype)
            # # Sample time
            # t = torch.rand(z_2.shape[0], 1, device=device, dtype=flow.dtype)
            # # Create the interpolated points on the path from noise (z_0) to target data (z_2)
            # z_t = (1 - t) * z_0 + t * z_2
            # # The target velocity is the vector from noise to the target data
            # target_velocity = z_2 - z_0
            # # Forward pass through the *conditional* model
            # predicted_velocity = flow(z_t, t, z_1) # Note the third argument: z_1 is the condition

            # ## cylcic loss
            # n_steps = 10
            # time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
            # z12 = z_1.clone()
            # for step in range(n_steps):
            #     z12 = flow_12.step(z12, time_steps[step], time_steps[step + 1])
            # z = encoder_2(decoder_2(z12))
            # for step in range(n_steps):
            #     z = flow_21.step(z, time_steps[step], time_steps[step + 1])
            # y_cycled_1 = decoder_1(z)
            # # repeat fro 2 to 1
            # z21 = z_2.clone()
            # for step in range(n_steps):
            #     z21 = flow_21.step(z21, time_steps[step], time_steps[step + 1])
            # z = encoder_1(decoder_1(z21))
            # for step in range(n_steps):
            #     z = flow_12.step(z, time_steps[step], time_steps[step + 1])
            # y_cycled_2 = decoder_2(z)
            # loss_cycle = torch.mean(torch.sum((y_cycled_1 - X_1)**2,dim=1)) + torch.mean(torch.sum((y_cycled_2 - X_2)**2,dim=1))

            loss = model_params['flow_lambda'] * flow_loss + model_params['conditional_flow_lambda']*dist
            loss.backward(retain_graph=True)
            # loss = loss_fn(predicted_velocity, target_velocity)
            optimizer.step()

            batch_losses.append(loss.item())
            batch_flow_losses.append(flow_loss.item())
            batch_dist_losses.append(dist.item())

        scheduler.step()
        outString = 'Epoch={:.0f}/{:.0f}'.format(e+1,NUM_EPOCHS)
        outString += ', distance loss={:.4f}'.format(np.nanmean(batch_dist_losses))
        outString += ', flow loss={:.4f}'.format(np.nanmean(batch_flow_losses))
        # outString += ', r cycle_1={:.4f}'.format(rcycle_1.item())
        # outString += ', r cycle_2={:.4f}'.format(rcycle_2.item())
        # outString += ', flow_loss_12={:.4f}'.format(flow_loss_12.item())
        # outString += ', flow_loss_21={:.4f}'.format(flow_loss_21.item())
        outString += ', loss={:.4f}'.format(np.nanmean(batch_losses))

        all_losses.append(np.nanmean(batch_losses))
        all_flow_losses.append(np.nanmean(batch_flow_losses))
        all_dist_losses.append(np.nanmean(batch_dist_losses))

        # Logging
        # if (e % 250 == 0) or (e + 1 == NUM_EPOCHS):
        print2log(outString)
        
    # from matplotlib import pyplot as plt
    # plt.plot(np.arange(NUM_EPOCHS),all_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('total loss')
    # plt.show()

    # plt.plot(np.arange(NUM_EPOCHS),all_flow_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('flow loss')
    # plt.show()

    # plt.plot(np.arange(NUM_EPOCHS),all_dist_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('distance loss')
    # plt.show()
    # evaluate
    flow.eval()
    with torch.no_grad():
        # translation results
        if pairs_train is not None:
            x_1_equivalent = x_1_train[pairs_train,:].double()
            x_2_equivalent = x_2_train[pairs_train,:].double()
            # # first translate system 1 to 2
            # z_latent_base_1_equivalent  = encoder_1(x_1_equivalent)
            # z_latent_base_2_equivalent  = encoder_2(x_2_equivalent)
            z_latent_base_1_equivalent  = z_1_train[pairs_train,:].double()
            z_latent_base_2_equivalent  = z_2_train[pairs_train,:].double()
            # cosine of paired conditions
            cosineLoss = torch.nn.functional.cosine_similarity(z_latent_base_1_equivalent,z_latent_base_2_equivalent,dim=-1).mean()
            ## flow step
            if tanslation_direction == '1 to 2':
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                for step in range(n_steps):
                    z_latent_base_1_equivalent = flow.step(z_latent_base_1_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_2_equivalent = decoder_2(z_latent_base_1_equivalent).detach()
                pearson_translation = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()
            else:
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                # second translate system 2 to 1
                for step in range(n_steps):
                    z_latent_base_2_equivalent = flow.step(z_latent_base_2_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_1_equivalent = decoder_1(z_latent_base_2_equivalent).detach()
                pearson_translation = pearson_r(x_hat_1_equivalent.flatten(), x_1_equivalent.flatten()).detach().cpu().numpy()

            # # 1. The source vector `z_1_equivalent` is now the *condition*. It remains fixed.
            # z_1_condition = z_latent_base_1_equivalent
            # # 2. We start the flow from random noise in the *target* space (space 2).
            # z_t = torch.randn_like(z_latent_base_2_equivalent, device=device, dtype=flow.dtype)
            # # 3. Solve the ODE over time
            # n_steps = 10 # Or any number of steps you prefer for evaluation
            # time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=flow.dtype)
            # for step in range(n_steps):
            #     t_start, t_end = time_steps[step], time_steps[step + 1]
            #     # Call the model's step function, providing z_t, time, and the condition
            #     z_t = flow.step(z_t, t_start, t_end, z_1_condition)
            # # `z_t` is now the generated latent vector in space 2
            # z_hat_2_equivalent = z_t
            # # 4. Decode the generated latent vector
            # x_hat_2_equivalent = decoder_2(z_hat_2_equivalent).detach()
            # # 5. Calculate the Pearson correlation coefficient
            # pearson_translation = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()

            print2log('cosine of trained paired conditions: {:.4f}'.format(cosineLoss.item()))
            cosineLoss = cosineLoss.item()
        else:
            pearson_1_to_2 = np.nan
            pearson_2_to_1 = np.nan
            cosineLoss = np.nan

    return (pearson_translation,cosineLoss,flow)

def validate_flowMatch_fold(device, x_1_test, x_2_test,
                              decoder_1, decoder_2, encoder_1, encoder_2,flow,
                              pairs_val=None,tanslation_direction='1 to 2'):
    # Evaluation mode
    encoder_1.eval()
    encoder_2.eval()
    decoder_1.eval()
    decoder_2.eval()
    flow.eval()
    
    with torch.no_grad():
        # Prepare test data
        x_1 = x_1_test.double().to(device)
        x_2 = x_2_test.double().to(device)
        # Generate latent variables
        z_latent_base_1 = encoder_1(x_1)
        z_latent_base_2 = encoder_2(x_2)
        # reconstruction results
        y_pred_1 = decoder_1(z_latent_base_1)
        y_pred_2 = decoder_2(z_latent_base_2)
        # evaluate pearson correlation (pearson_r) of reconstruction
        r_1 = pearson_r(y_pred_1.flatten(), x_1.flatten()).detach().cpu().numpy()
        r_2 = pearson_r(y_pred_2.flatten(), x_2.flatten()).detach().cpu().numpy()

        # translation results
        if pairs_val is not None:
            x_1_equivalent = x_1[pairs_val,:]
            x_2_equivalent = x_2[pairs_val,:]
            # first translate system 1 to 2
            z_latent_base_1_equivalent  = encoder_1(x_1_equivalent)
            z_latent_base_2_equivalent  = encoder_2(x_2_equivalent)
            cosineLoss = torch.nn.functional.cosine_similarity(z_latent_base_1_equivalent,z_latent_base_2_equivalent,dim=-1).mean()
            n_steps = 10
            time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
            
            # # --- FIX: Use conditional generation for translation ---
            # if tanslation_direction == '1 to 2':
            #     # 1. The source vector is the condition.
            #     z_1_condition = z_latent_base_1_equivalent

            #     # 2. Start from random noise in the target space.
            #     z_t = torch.randn_like(z_latent_base_2_equivalent, device=device, dtype=flow.dtype)

            #     # 3. Solve the ODE over time, providing the condition at each step.
            #     for step in range(n_steps):
            #         t_start, t_end = time_steps[step], time_steps[step + 1]
            #         z_t = flow.step(z_t, t_start, t_end, z_1_condition)

            #     # The final z_t is the generated latent vector.
            #     z_hat_2_equivalent = z_t
                
            #     # 4. Decode and evaluate.
            #     x_hat_2_equivalent = decoder_2(z_hat_2_equivalent).detach()
            #     pearson_translation = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()
            
            # # Note: Translating 2 to 1 would require a separate, second model trained for that direction.
            # # The logic below is for an unconditional flow and would need to be adapted for a second conditional model.
            # else: # tanslation_direction == '2 to 1'
            #     # THIS PART IS NOT SUPPORTED BY THE CURRENT SINGLE-DIRECTIONAL MODEL
            #     # To implement this, you would need a second model, flow_21, trained
            #     # to generate z1 conditioned on z2. The logic would be symmetric.
            #     # For now, this will raise an error if called.
            #     pearson_translation = np.nan # Placeholder
            if tanslation_direction == '1 to 2':
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                for step in range(n_steps):
                    z_latent_base_1_equivalent = flow.step(z_latent_base_1_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_2_equivalent = decoder_2(z_latent_base_1_equivalent).detach()
                pearson_translation = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()
            else:
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                # second translate system 2 to 1
                for step in range(n_steps):
                    z_latent_base_2_equivalent = flow.step(z_latent_base_2_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_1_equivalent = decoder_1(z_latent_base_2_equivalent).detach()
                pearson_translation = pearson_r(x_hat_1_equivalent.flatten(), x_1_equivalent.flatten()).detach().cpu().numpy()

            # also caclulate cosize similarity between z_latent_base_1_equivalent and z_latent_base_2_equivalent
            print2log('cosine of validation paired conditions: {:.4f}'.format(cosineLoss.item()))
            cosineLoss = cosineLoss.item()
            
        else:
            pearson_translation = np.nan
            cosineLoss = np.nan
    
    return pearson_translation, r_1, r_2,cosineLoss

def train_AE_fold(model_params, device, x_train,
                    decoder, encoder,
                    bs:int, NUM_EPOCHS:int,
                    evaluate:bool=True):
    # Combine parameters and create optimizers/schedulers
    allParams = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    # Get dataset sizes
    N = x_train.shape[0]
    batch_pearsons = []
    batch_r2s = []
    batch_MSEs = []
    batch_losses = []
    # Training loop
    for e in range(NUM_EPOCHS):        
        decoder.train()
        encoder.train()
        # Generate training batches
        trainloader = getSamples(N, bs)
        # Iterate through batches
        for j in range(len(trainloader)):
            dataIndex = trainloader[j]            
            X = x_train[dataIndex, :].double().to(device)
            optimizer.zero_grad()
            # encode
            z = encoder(X)
            # decode
            y_pred = decoder(z)
            # calculate loss
            fitLoss = torch.mean(torch.sum((y_pred - X)**2,dim=1))
            L2Loss = decoder.L2Regularization(model_params['dec_l2_reg']) + encoder.L2Regularization(model_params['enc_l2_reg'])
            loss = fitLoss + L2Loss
            loss.backward()
            optimizer.step()
            # Get performance metrics
            pearson = torch.nanmean(pearson_r(y_pred.detach(), X.detach()))
            r2 = r_square(y_pred.detach().flatten(), X.detach().flatten())
            mse = torch.mean(torch.mean((y_pred.detach() - X.detach())**2,dim=1))
            batch_pearsons.append(pearson.item())
            batch_r2s.append(r2.item())
            batch_MSEs.append(mse.item())
            batch_losses.append(loss.item())
        
        scheduler.step()
        outString = 'Epoch={:.0f}/{:.0f}'.format(e+1,NUM_EPOCHS)
        outString += ', R2={:.4f}'.format(np.nanmean(batch_r2s))
        outString += ', pearson={:.4f}'.format(np.nanmean(batch_pearsons))
        outString += ', MSE={:.4f}'.format(np.nanmean(batch_MSEs))
        outString += ', loss={:.4f}'.format(np.nanmean(batch_losses))

        # Logging
        if (e % 250 == 0) or (e + 1 == NUM_EPOCHS):
            print2log(outString)
        
    # evaluate
    if evaluate:
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            y_pred = decoder(encoder(x_train.double().to(device)))
            pear = pearson_r(y_pred.flatten().detach(), x_train.flatten().double().to(device)).cpu().numpy()
    else:
        pear = None

    return (pear,decoder, encoder)

def train_decoder_fold(model_params, device, x_train, x_train_aligned,
                    decoder,
                    bs:int, NUM_EPOCHS:int):
    # Combine parameters and create optimizers/schedulers
    optimizer = torch.optim.Adam(decoder.parameters(), lr=model_params['lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    # Get dataset sizes
    N = x_train.shape[0]
    batch_pearsons = []
    batch_r2s = []
    batch_MSEs = []
    batch_losses = []
    # Training loop
    for e in range(NUM_EPOCHS):        
        decoder.train()
        # Generate training batches
        trainloader = getSamples(N, bs)
        # Iterate through batches
        for j in range(len(trainloader)):
            dataIndex = trainloader[j]            
            X = x_train_aligned[dataIndex, :].double().to(device)
            Y = x_train[dataIndex, :].double().to(device)
            optimizer.zero_grad()
            # decode
            y_pred = decoder(X)
            # calculate loss
            fitLoss = torch.mean(torch.sum((y_pred - Y)**2,dim=1))
            L2Loss = decoder.L2Regularization(model_params['dec_l2_reg'])
            loss = fitLoss + L2Loss
            loss.backward()
            optimizer.step()
            # Get performance metrics
            pearson = torch.nanmean(pearson_r(y_pred.detach(), Y.detach()))
            r2 = r_square(y_pred.detach().flatten(), Y.detach().flatten())
            mse = torch.mean(torch.mean((y_pred.detach() - Y.detach())**2,dim=1))
            batch_pearsons.append(pearson.item())
            batch_r2s.append(r2.item())
            batch_MSEs.append(mse.item())
            batch_losses.append(loss.item())
        
        scheduler.step()
        outString = 'Epoch={:.0f}/{:.0f}'.format(e+1,NUM_EPOCHS)
        outString += ', R2={:.4f}'.format(np.nanmean(batch_r2s))
        outString += ', pearson={:.4f}'.format(np.nanmean(batch_pearsons))
        outString += ', MSE={:.4f}'.format(np.nanmean(batch_MSEs))
        outString += ', loss={:.4f}'.format(np.nanmean(batch_losses))

        # Logging
        if (e % 250 == 0) or (e + 1 == NUM_EPOCHS):
            print2log(outString)
        
    # evaluate
    decoder.eval()
    with torch.no_grad():
        y_pred = decoder(x_train_aligned.double().to(device))
        pear = pearson_r(y_pred.flatten().detach(), x_train.flatten().double().to(device)).cpu().numpy()

    return (pear,decoder)

def train_bidirectional_flowMatch_fold(model_params, device, x_1_train, x_2_train,z_1_train,z_2_train,
                    decoder_1, decoder_2,flow,
                    bs_1:int, bs_2:int, NUM_EPOCHS:int,
                    pairs_train=None,
                    tanslation_direction = '1 to 2'):
    # Combine parameters and create optimizers/schedulers
    allParams = list(flow.parameters())
    
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    flow_loss_fn = torch.nn.MSELoss()
    decoder_1.eval()
    decoder_2.eval()

    # Get dataset sizes
    N_1 = x_1_train.shape[0]
    N_2 = x_2_train.shape[0]
    all_losses = []
    all_flow_losses = []
    all_dist_losses = []
    # Training loop
    # loss_fn = torch.nn.MSELoss()
    for e in range(NUM_EPOCHS):        
        flow.train()
        # Generate training batches
        trainloader_1 = getSamples(N_1, bs_1)
        trainloader_2 = getSamples(N_2, bs_2)
        maxLen = max(len(trainloader_1), len(trainloader_2))
        # Pad shorter trainloader if needed
        if len(trainloader_1) < maxLen:
            while len(trainloader_1) < maxLen:
                trainloader_1 += getSamples(N_1, bs_1)[:maxLen - len(trainloader_1)]
        if len(trainloader_2) < maxLen:
            while len(trainloader_2) < maxLen:
                trainloader_2 += getSamples(N_2, bs_2)[:maxLen - len(trainloader_2)]
        # Iterate through batches
        batch_losses = []
        batch_flow_losses = []
        batch_dist_losses = []
        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]
            # Get data for the batch
            X_1 = x_1_train[dataIndex_1, :]#.cpu().numpy()#.double().to(device)
            X_2 = x_2_train[dataIndex_2, :]#.cpu().numpy()#.double().to(device)
            z_1 = z_1_train[dataIndex_1, :]#.double().to(device)
            z_2 = z_2_train[dataIndex_2, :]#.double().to(device)
            # X_1_aligned, X_2_aligned,  _,_ = transact_align_gpu(
            #     X_1,        # source → will become Z_source
            #     X_2,        # target → will become Z_target
            #     n_src_pcs=75,
            #     n_tgt_pcs=75,
            #     n_pv=30,
            #     kernel='rbf',
            #     gamma=5e-4      # or whatever you tuned
            # )
            X_2_aligned, X_1_aligned, _, _ = transact_align_gpu(
                X_2,        # source → will become Z_source
                X_1,        # target → will become Z_target
                n_src_pcs=75,
                n_tgt_pcs=75,
                n_pv=30,
                kernel='rbf',
                gamma=5e-4      # or whatever you tuned
            )
            # X_1_aligned = torch.from_numpy(X_1_aligned).double().to(device)
            # X_2_aligned = torch.from_numpy(X_2_aligned).double().to(device)
            X_1_aligned = X_1_aligned  - X_1_aligned.mean(0)
            X_2_aligned = X_2_aligned  - X_2_aligned.mean(0)
            C = X_1_aligned @ X_2_aligned.T
            C = C/C.max()
            optimizer.zero_grad()
            if tanslation_direction == '1 to 2':
                # Translate z1 to z2 with flow
                if z_1.shape[0] == z_2.shape[0]:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_1 + t * z_2
                    dz_t = z_2 - z_1
                elif z_1.shape[0] > z_2.shape[0]:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_1[:z_2.shape[0]] + t * z_2
                    dz_t = z_2 - z_1[:z_2.shape[0]]
                else:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_1 + t * z_2[:z_1.shape[0]]
                    dz_t = z_2[:z_1.shape[0]] - z_1
                flow_loss = flow_loss_fn(flow(z_t, t), dz_t)
                ## Conditioned based on initial correlation
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                z12 = z_1.clone()
                for step in range(n_steps):
                    z12 = flow.step(z12, time_steps[step], time_steps[step + 1])
                dist = torch.cdist(z12, z_2) * torch.nn.functional.relu(C)
                dist = torch.sum(dist)
            else:
                # Translate now z2 to z1
                if z_1.shape[0] == z_2.shape[0]:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_2 + t * z_1
                    dz_t = z_1 - z_2
                elif z_2.shape[0] > z_1.shape[0]:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_2[:z_1.shape[0]] + t * z_1
                    dz_t = z_1 - z_2[:z_1.shape[0]]
                else:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_2 + t * z_1[:z_2.shape[0]]
                    dz_t = z_1[:z_2.shape[0]] - z_2
                flow_loss = flow_loss_fn(flow(z_t, t), dz_t)
                ## Conditioned based on initial correlation
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                z21 = z_2.clone()
                for step in range(n_steps):
                    z21 = flow.step(z21, time_steps[step], time_steps[step + 1])
                dist = torch.cdist(z_1, z21) * torch.nn.functional.relu(C)
                dist = torch.sum(dist)
            # # Sample from the prior distribution in the *target* space (R^m2)
            # z_0 = torch.randn_like(z_2, device=device, dtype=flow.dtype)
            # # Sample time
            # t = torch.rand(z_2.shape[0], 1, device=device, dtype=flow.dtype)
            # # Create the interpolated points on the path from noise (z_0) to target data (z_2)
            # z_t = (1 - t) * z_0 + t * z_2
            # # The target velocity is the vector from noise to the target data
            # target_velocity = z_2 - z_0
            # # Forward pass through the *conditional* model
            # predicted_velocity = flow(z_t, t, z_1) # Note the third argument: z_1 is the condition

            # ## cylcic loss
            # n_steps = 10
            # time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
            # z12 = z_1.clone()
            # for step in range(n_steps):
            #     z12 = flow_12.step(z12, time_steps[step], time_steps[step + 1])
            # z = encoder_2(decoder_2(z12))
            # for step in range(n_steps):
            #     z = flow_21.step(z, time_steps[step], time_steps[step + 1])
            # y_cycled_1 = decoder_1(z)
            # # repeat fro 2 to 1
            # z21 = z_2.clone()
            # for step in range(n_steps):
            #     z21 = flow_21.step(z21, time_steps[step], time_steps[step + 1])
            # z = encoder_1(decoder_1(z21))
            # for step in range(n_steps):
            #     z = flow_12.step(z, time_steps[step], time_steps[step + 1])
            # y_cycled_2 = decoder_2(z)
            # loss_cycle = torch.mean(torch.sum((y_cycled_1 - X_1)**2,dim=1)) + torch.mean(torch.sum((y_cycled_2 - X_2)**2,dim=1))

            loss = model_params['flow_lambda'] * flow_loss + model_params['conditional_flow_lambda']*dist
            loss.backward(retain_graph=True)
            # loss = loss_fn(predicted_velocity, target_velocity)
            optimizer.step()

            batch_losses.append(loss.item())
            batch_flow_losses.append(flow_loss.item())
            batch_dist_losses.append(dist.item())

        scheduler.step()
        outString = 'Epoch={:.0f}/{:.0f}'.format(e+1,NUM_EPOCHS)
        outString += ', distance loss={:.4f}'.format(np.nanmean(batch_dist_losses))
        outString += ', flow loss={:.4f}'.format(np.nanmean(batch_flow_losses))
        # outString += ', r cycle_1={:.4f}'.format(rcycle_1.item())
        # outString += ', r cycle_2={:.4f}'.format(rcycle_2.item())
        # outString += ', flow_loss_12={:.4f}'.format(flow_loss_12.item())
        # outString += ', flow_loss_21={:.4f}'.format(flow_loss_21.item())
        outString += ', loss={:.4f}'.format(np.nanmean(batch_losses))

        all_losses.append(np.nanmean(batch_losses))
        all_flow_losses.append(np.nanmean(batch_flow_losses))
        all_dist_losses.append(np.nanmean(batch_dist_losses))

        # Logging
        if (e % 250 == 0) or (e + 1 == NUM_EPOCHS):
            print2log(outString)
        
    # from matplotlib import pyplot as plt
    # plt.plot(np.arange(NUM_EPOCHS),all_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('total loss')
    # plt.show()

    # plt.plot(np.arange(NUM_EPOCHS),all_flow_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('flow loss')
    # plt.show()

    # plt.plot(np.arange(NUM_EPOCHS),all_dist_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('distance loss')
    # plt.show()
    # evaluate
    flow.eval()
    with torch.no_grad():
        # translation results
        if pairs_train is not None:
            x_1_equivalent = x_1_train[pairs_train,:].double()
            x_2_equivalent = x_2_train[pairs_train,:].double()
            # # first translate system 1 to 2
            # z_latent_base_1_equivalent  = encoder_1(x_1_equivalent)
            # z_latent_base_2_equivalent  = encoder_2(x_2_equivalent)
            z_latent_base_1_equivalent  = z_1_train[pairs_train,:].double()
            z_latent_base_2_equivalent  = z_2_train[pairs_train,:].double()
            # cosine of paired conditions
            cosineLoss = torch.nn.functional.cosine_similarity(z_latent_base_1_equivalent,z_latent_base_2_equivalent,dim=-1).mean()
            ## flow step
            if tanslation_direction == '1 to 2':
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                for step in range(n_steps):
                    z_latent_base_1_equivalent = flow.step(z_latent_base_1_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_2_equivalent = decoder_2(z_latent_base_1_equivalent).detach()
                pearson_translation = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()
            else:
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                # second translate system 2 to 1
                for step in range(n_steps):
                    z_latent_base_2_equivalent = flow.step(z_latent_base_2_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_1_equivalent = decoder_1(z_latent_base_2_equivalent).detach()
                pearson_translation = pearson_r(x_hat_1_equivalent.flatten(), x_1_equivalent.flatten()).detach().cpu().numpy()

            # # 1. The source vector `z_1_equivalent` is now the *condition*. It remains fixed.
            # z_1_condition = z_latent_base_1_equivalent
            # # 2. We start the flow from random noise in the *target* space (space 2).
            # z_t = torch.randn_like(z_latent_base_2_equivalent, device=device, dtype=flow.dtype)
            # # 3. Solve the ODE over time
            # n_steps = 10 # Or any number of steps you prefer for evaluation
            # time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=flow.dtype)
            # for step in range(n_steps):
            #     t_start, t_end = time_steps[step], time_steps[step + 1]
            #     # Call the model's step function, providing z_t, time, and the condition
            #     z_t = flow.step(z_t, t_start, t_end, z_1_condition)
            # # `z_t` is now the generated latent vector in space 2
            # z_hat_2_equivalent = z_t
            # # 4. Decode the generated latent vector
            # x_hat_2_equivalent = decoder_2(z_hat_2_equivalent).detach()
            # # 5. Calculate the Pearson correlation coefficient
            # pearson_translation = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()

            print2log('cosine of trained paired conditions: {:.4f}'.format(cosineLoss.item()))
            cosineLoss = cosineLoss.item()
        else:
            pearson_1_to_2 = np.nan
            pearson_2_to_1 = np.nan
            cosineLoss = np.nan

    return (pearson_translation,cosineLoss,flow)

def validate_bidirectional_flowMatch_fold(device, x_1_test, x_2_test,
                              decoder_1, decoder_2, encoder_1, encoder_2,flow,
                              pairs_val=None,tanslation_direction='1 to 2'):
    # Evaluation mode
    encoder_1.eval()
    encoder_2.eval()
    decoder_1.eval()
    decoder_2.eval()
    flow.eval()
    
    with torch.no_grad():
        # Prepare test data
        x_1 = x_1_test.double().to(device)
        x_2 = x_2_test.double().to(device)
        # Generate latent variables
        z_latent_base_1 = encoder_1(x_1)
        z_latent_base_2 = encoder_2(x_2)
        # reconstruction results
        y_pred_1 = decoder_1(z_latent_base_1)
        y_pred_2 = decoder_2(z_latent_base_2)
        # evaluate pearson correlation (pearson_r) of reconstruction
        r_1 = pearson_r(y_pred_1.flatten(), x_1.flatten()).detach().cpu().numpy()
        r_2 = pearson_r(y_pred_2.flatten(), x_2.flatten()).detach().cpu().numpy()

        # translation results
        if pairs_val is not None:
            x_1_equivalent = x_1[pairs_val,:]
            x_2_equivalent = x_2[pairs_val,:]
            # first translate system 1 to 2
            z_latent_base_1_equivalent  = encoder_1(x_1_equivalent)
            z_latent_base_2_equivalent  = encoder_2(x_2_equivalent)
            cosineLoss = torch.nn.functional.cosine_similarity(z_latent_base_1_equivalent,z_latent_base_2_equivalent,dim=-1).mean()
            n_steps = 10
            time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
            
            # # --- FIX: Use conditional generation for translation ---
            # if tanslation_direction == '1 to 2':
            #     # 1. The source vector is the condition.
            #     z_1_condition = z_latent_base_1_equivalent

            #     # 2. Start from random noise in the target space.
            #     z_t = torch.randn_like(z_latent_base_2_equivalent, device=device, dtype=flow.dtype)

            #     # 3. Solve the ODE over time, providing the condition at each step.
            #     for step in range(n_steps):
            #         t_start, t_end = time_steps[step], time_steps[step + 1]
            #         z_t = flow.step(z_t, t_start, t_end, z_1_condition)

            #     # The final z_t is the generated latent vector.
            #     z_hat_2_equivalent = z_t
                
            #     # 4. Decode and evaluate.
            #     x_hat_2_equivalent = decoder_2(z_hat_2_equivalent).detach()
            #     pearson_translation = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()
            
            # # Note: Translating 2 to 1 would require a separate, second model trained for that direction.
            # # The logic below is for an unconditional flow and would need to be adapted for a second conditional model.
            # else: # tanslation_direction == '2 to 1'
            #     # THIS PART IS NOT SUPPORTED BY THE CURRENT SINGLE-DIRECTIONAL MODEL
            #     # To implement this, you would need a second model, flow_21, trained
            #     # to generate z1 conditioned on z2. The logic would be symmetric.
            #     # For now, this will raise an error if called.
            #     pearson_translation = np.nan # Placeholder
            if tanslation_direction == '1 to 2':
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                for step in range(n_steps):
                    z_latent_base_1_equivalent = flow.step(z_latent_base_1_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_2_equivalent = decoder_2(z_latent_base_1_equivalent).detach()
                pearson_translation = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()
            else:
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                # second translate system 2 to 1
                for step in range(n_steps):
                    z_latent_base_2_equivalent = flow.step(z_latent_base_2_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_1_equivalent = decoder_1(z_latent_base_2_equivalent).detach()
                pearson_translation = pearson_r(x_hat_1_equivalent.flatten(), x_1_equivalent.flatten()).detach().cpu().numpy()

            # also caclulate cosize similarity between z_latent_base_1_equivalent and z_latent_base_2_equivalent
            print2log('cosine of validation paired conditions: {:.4f}'.format(cosineLoss.item()))
            cosineLoss = cosineLoss.item()
            
        else:
            pearson_translation = np.nan
            cosineLoss = np.nan
    
    return pearson_translation, r_1, r_2,cosineLoss

#### Train flowMatch giving it a similarity matrix to use for mapping ####
def train_flowMatch_withpairs_fold(model_params, device, x_1_train, x_2_train,z_1_train,z_2_train,
                                   allembs1, allembs2, trainInfo_1, trainInfo_2,trainInfo_paired,
                    encoder_1, encoder_2, decoder_1, decoder_2,flow,
                    bs_1:int, bs_2:int, bs_paired:int, NUM_EPOCHS:int,
                    pairs_train=None,
                    tanslation_direction = '1 to 2'):
    # Combine parameters and create optimizers/schedulers
    allParams = list(flow.parameters())
    
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    flow_loss_fn = torch.nn.MSELoss()
    # encoder_1.eval()
    decoder_1.eval()
    # encoder_2.eval()
    decoder_2.eval()

    # Get dataset sizes
    N_paired = len(trainInfo_paired)
    N_1 = len(trainInfo_1)
    N_2 = len(trainInfo_2)
    N = N_1
    if N_2>N:
        N=N_2
    all_losses = []
    all_flow_losses = []
    all_dist_losses = []
    # Training loop
    # loss_fn = torch.nn.MSELoss()
    for e in range(NUM_EPOCHS):        
        flow.train()
        # Generate training batches
        trainloader_1 = getSamples(N_1, bs_1)
        len_1 = len(trainloader_1)
        trainloader_2 = getSamples(N_2, bs_2)
        len_2 = len(trainloader_2)
        trainloader_paired = getSamples(N_paired, bs_paired)
        len_paired = len(trainloader_paired)
        lens = [len_1,len_2,len_paired]
        maxLen = np.max(lens)
        # Pad shorter trainloader if needed
        if len(trainloader_1) < maxLen:
            while len(trainloader_1) < maxLen:
                trainloader_1 += getSamples(N_1, bs_1)[:maxLen - len(trainloader_1)]
        if len(trainloader_2) < maxLen:
            while len(trainloader_2) < maxLen:
                trainloader_2 += getSamples(N_2, bs_2)[:maxLen - len(trainloader_2)]
        if len(trainloader_paired) < maxLen:
            while len(trainloader_paired) < maxLen:
                trainloader_paired += getSamples(N_paired, bs_paired)[:maxLen - len(trainloader_paired)]
        # Iterate through batches
        batch_losses = []
        batch_flow_losses = []
        batch_dist_losses = []
        for j in range(maxLen):
            # dataIndex_1 = _fix_batch_size(trainloader_1[j],bs_1,N_1)
            # dataIndex_2 = _fix_batch_size(trainloader_2[j],bs_1,N_2)
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]
            dataIndex_paired = trainloader_paired[j]

            # Get batch
            df_pairs = trainInfo_paired.iloc[dataIndex_paired,:]
            df_1 = trainInfo_1.iloc[dataIndex_1,:]
            df_2 = trainInfo_2.iloc[dataIndex_2,:]
            # Now concatenate; these have the exact expected row counts:
            z_1 = torch.tensor(np.concatenate((allembs1.loc[df_pairs['sig_id.x']].values,
                                                 allembs1.loc[df_1.sig_id].values))).double().to(device)
            z_2 = torch.tensor(np.concatenate((allembs2.loc[df_pairs['sig_id.y']].values,
                                           allembs2.loc[df_2.sig_id].values))).double().to(device)
            
            conditions = np.concatenate((df_pairs.conditionId.values,
                                            df_1.conditionId.values,
                                            df_pairs.conditionId.values,
                                            df_2.conditionId.values))
            size = conditions.size
            conditions = conditions.reshape(size,1)
            conditions = conditions == conditions.transpose()
            conditions = conditions*1
            # conditions = conditions[0:(z_1.shape[0]+1),(z_1.shape[0]+1):(z_1.shape[0]+z_2.shape[0]+1)]
            C = torch.tensor(conditions,dtype=torch.double).to(device).detach()

            # print2log(z_1.shape)
            # print2log(z_2.shape)
            # print2log(C.shape)

            optimizer.zero_grad()
            if tanslation_direction == '1 to 2':
                # Translate z1 to z2 with flow
                if z_1.shape[0] == z_2.shape[0]:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_1 + t * z_2
                    dz_t = z_2 - z_1
                elif z_1.shape[0] > z_2.shape[0]:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_1[:z_2.shape[0]] + t * z_2
                    dz_t = z_2 - z_1[:z_2.shape[0]]
                else:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_1 + t * z_2[:z_1.shape[0]]
                    dz_t = z_2[:z_1.shape[0]] - z_1
                flow_loss = flow_loss_fn(flow(z_t, t), dz_t)
                ## Conditioned based on initial correlation
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                z12 = z_1.clone()
                for step in range(n_steps):
                    z12 = flow.step(z12, time_steps[step], time_steps[step + 1])
                dist = torch.cdist(torch.cat((z12, z12), 0), torch.cat((z12, z_2), 0)) * C
                dist = torch.sum(dist)
            else:
                # Translate now z2 to z1
                if z_1.shape[0] == z_2.shape[0]:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_2 + t * z_1
                    dz_t = z_1 - z_2
                elif z_2.shape[0] > z_1.shape[0]:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_2[:z_1.shape[0]] + t * z_1
                    dz_t = z_1 - z_2[:z_1.shape[0]]
                else:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_2 + t * z_1[:z_2.shape[0]]
                    dz_t = z_1[:z_2.shape[0]] - z_2
                flow_loss = flow_loss_fn(flow(z_t, t), dz_t)
                ## Conditioned based on initial correlation
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                z21 = z_2.clone()
                for step in range(n_steps):
                    z21 = flow.step(z21, time_steps[step], time_steps[step + 1])
                dist = torch.cdist(torch.cat((z_1, z21), 0), torch.cat((z21, z21), 0)) * C
                dist = torch.sum(dist)
            loss = model_params['flow_lambda'] * flow_loss + model_params['conditional_flow_lambda']*dist
            loss.backward(retain_graph=True)
            # loss = loss_fn(predicted_velocity, target_velocity)
            optimizer.step()

            batch_losses.append(loss.item())
            batch_flow_losses.append(flow_loss.item())
            batch_dist_losses.append(dist.item())

        scheduler.step()
        outString = 'Epoch={:.0f}/{:.0f}'.format(e+1,NUM_EPOCHS)
        outString += ', distance loss={:.4f}'.format(np.nanmean(batch_dist_losses))
        outString += ', flow loss={:.4f}'.format(np.nanmean(batch_flow_losses))
        # outString += ', r cycle_1={:.4f}'.format(rcycle_1.item())
        # outString += ', r cycle_2={:.4f}'.format(rcycle_2.item())
        # outString += ', flow_loss_12={:.4f}'.format(flow_loss_12.item())
        # outString += ', flow_loss_21={:.4f}'.format(flow_loss_21.item())
        outString += ', loss={:.4f}'.format(np.nanmean(batch_losses))

        all_losses.append(np.nanmean(batch_losses))
        all_flow_losses.append(np.nanmean(batch_flow_losses))
        all_dist_losses.append(np.nanmean(batch_dist_losses))

        # Logging
        if (e % 250 == 0) or (e + 1 == NUM_EPOCHS):
            print2log(outString)
    flow.eval()
    with torch.no_grad():
        # translation results
        if pairs_train is not None:
            x_1_equivalent = x_1_train[pairs_train,:].double()
            x_2_equivalent = x_2_train[pairs_train,:].double()
            # # first translate system 1 to 2
            z_latent_base_1_equivalent  = z_1_train[pairs_train,:].double()
            z_latent_base_2_equivalent  = z_2_train[pairs_train,:].double()
            # cosine of paired conditions
            cosineLoss = torch.nn.functional.cosine_similarity(z_latent_base_1_equivalent,z_latent_base_2_equivalent,dim=-1).mean()
            ## flow step
            if tanslation_direction == '1 to 2':
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                for step in range(n_steps):
                    z_latent_base_1_equivalent = flow.step(z_latent_base_1_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_2_equivalent = decoder_2(z_latent_base_1_equivalent).detach()
                pearson_translation = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()
            else:
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                # second translate system 2 to 1
                for step in range(n_steps):
                    z_latent_base_2_equivalent = flow.step(z_latent_base_2_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_1_equivalent = decoder_1(z_latent_base_2_equivalent).detach()
                pearson_translation = pearson_r(x_hat_1_equivalent.flatten(), x_1_equivalent.flatten()).detach().cpu().numpy()

            print2log('cosine of trained paired conditions: {:.4f}'.format(cosineLoss.item()))
            cosineLoss = cosineLoss.item()
        else:
            pearson_1_to_2 = np.nan
            pearson_2_to_1 = np.nan
            cosineLoss = np.nan

    return (pearson_translation,cosineLoss,flow)


def train_flowMatch_withSTRUCTURE_fold(model_params, device, x_1_train, x_2_train,z_1_train,z_2_train,
                    decoder_1, decoder_2,flow,
                    bs_1:int, bs_2:int, NUM_EPOCHS:int,
                    pairs_train=None,
                    tanslation_direction = '1 to 2'):
    # Combine parameters and create optimizers/schedulers
    allParams = list(flow.parameters())
    
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    flow_loss_fn = torch.nn.MSELoss()
    decoder_1.eval()
    decoder_2.eval()

    # Get dataset sizes
    N_1 = x_1_train.shape[0]
    N_2 = x_2_train.shape[0]
    all_losses = []
    all_flow_losses = []
    all_structure_losses = []
    # Training loop
    # loss_fn = torch.nn.MSELoss()
    for e in range(NUM_EPOCHS):        
        flow.train()
        # Generate training batches
        trainloader_1 = getSamples(N_1, bs_1)
        trainloader_2 = getSamples(N_2, bs_2)
        maxLen = max(len(trainloader_1), len(trainloader_2))
        # Pad shorter trainloader if needed
        if len(trainloader_1) < maxLen:
            while len(trainloader_1) < maxLen:
                trainloader_1 += getSamples(N_1, bs_1)[:maxLen - len(trainloader_1)]
        if len(trainloader_2) < maxLen:
            while len(trainloader_2) < maxLen:
                trainloader_2 += getSamples(N_2, bs_2)[:maxLen - len(trainloader_2)]
        # Iterate through batches
        batch_losses = []
        batch_flow_losses = []
        batch_structure_losses = []
        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]
            # Get data for the batch
            z_1 = z_1_train[dataIndex_1, :]#.double().to(device)
            z_2 = z_2_train[dataIndex_2, :]#.double().to(device)
            optimizer.zero_grad()
            if tanslation_direction == '1 to 2':
                # Translate z1 to z2 with flow
                if z_1.shape[0] == z_2.shape[0]:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_1 + t * z_2
                    dz_t = z_2 - z_1
                elif z_1.shape[0] > z_2.shape[0]:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_1[:z_2.shape[0]] + t * z_2
                    dz_t = z_2 - z_1[:z_2.shape[0]]
                else:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_1 + t * z_2[:z_1.shape[0]]
                    dz_t = z_2[:z_1.shape[0]] - z_1
                flow_loss = flow_loss_fn(flow(z_t, t), dz_t)
                ## Conditioned based on initial correlation
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                z12 = z_1.clone()
                for step in range(n_steps):
                    z12 = flow.step(z12, time_steps[step], time_steps[step + 1])
                structure_loss = reg_structure(z12, z_2)
            else:
                # Translate now z2 to z1
                if z_1.shape[0] == z_2.shape[0]:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_2 + t * z_1
                    dz_t = z_1 - z_2
                elif z_2.shape[0] > z_1.shape[0]:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_2[:z_1.shape[0]] + t * z_1
                    dz_t = z_1 - z_2[:z_1.shape[0]]
                else:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_2 + t * z_1[:z_2.shape[0]]
                    dz_t = z_1[:z_2.shape[0]] - z_2
                flow_loss = flow_loss_fn(flow(z_t, t), dz_t)
                ## Conditioned based on initial correlation
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                z21 = z_2.clone()
                for step in range(n_steps):
                    z21 = flow.step(z21, time_steps[step], time_steps[step + 1])
                structure_loss = reg_structure(z21, z_1)
            
            loss = model_params['flow_lambda'] * flow_loss + model_params['conditional_flow_lambda']*structure_loss
            loss.backward(retain_graph=True)
            optimizer.step()

            batch_losses.append(loss.item())
            batch_flow_losses.append(flow_loss.item())
            batch_structure_losses.append(structure_loss.item())

        scheduler.step()
        outString = 'Epoch={:.0f}/{:.0f}'.format(e+1,NUM_EPOCHS)
        outString += ', distance loss={:.4f}'.format(np.nanmean(batch_structure_losses))
        outString += ', flow loss={:.4f}'.format(np.nanmean(batch_flow_losses))
        outString += ', loss={:.4f}'.format(np.nanmean(batch_losses))

        all_losses.append(np.nanmean(batch_losses))
        all_flow_losses.append(np.nanmean(batch_flow_losses))
        all_structure_losses.append(np.nanmean(batch_structure_losses))

        # Logging
        if (e % 250 == 0) or (e + 1 == NUM_EPOCHS):
            print2log(outString)
    flow.eval()
    with torch.no_grad():
        # translation results
        if pairs_train is not None:
            x_1_equivalent = x_1_train[pairs_train,:].double()
            x_2_equivalent = x_2_train[pairs_train,:].double()
            z_latent_base_1_equivalent  = z_1_train[pairs_train,:].double()
            z_latent_base_2_equivalent  = z_2_train[pairs_train,:].double()
            # cosine of paired conditions
            cosineLoss = torch.nn.functional.cosine_similarity(z_latent_base_1_equivalent,z_latent_base_2_equivalent,dim=-1).mean()
            ## flow step
            if tanslation_direction == '1 to 2':
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                for step in range(n_steps):
                    z_latent_base_1_equivalent = flow.step(z_latent_base_1_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_2_equivalent = decoder_2(z_latent_base_1_equivalent).detach()
                pearson_translation = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()
            else:
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                # second translate system 2 to 1
                for step in range(n_steps):
                    z_latent_base_2_equivalent = flow.step(z_latent_base_2_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_1_equivalent = decoder_1(z_latent_base_2_equivalent).detach()
                pearson_translation = pearson_r(x_hat_1_equivalent.flatten(), x_1_equivalent.flatten()).detach().cpu().numpy()
            print2log('cosine of trained paired conditions: {:.4f}'.format(cosineLoss.item()))
            cosineLoss = cosineLoss.item()
        else:
            pearson_1_to_2 = np.nan
            pearson_2_to_1 = np.nan
            cosineLoss = np.nan

    return (pearson_translation,cosineLoss,flow)


### Train FM using both existing pairs and pre-alignment similarity ###
def train_GeneralFM_fold(model_params, device, 
                         x_1_train, x_2_train,
                         z_1_train, z_2_train,
                         allX,
                         allembs1, allembs2, 
                         trainInfo_1, trainInfo_2,trainInfo_paired,
                         decoder_1, decoder_2,flow,
                         bs_1:int, bs_2:int, bs_paired:int, NUM_EPOCHS:int,
                         similarity_agregation = 'max', 
                         pairs_train=None,
                         tanslation_direction = '1 to 2'):
    # Validate similarity_agregation parameter
    if similarity_agregation not in ['mean', 'max', 'sum']:
        raise ValueError(f"similarity_agregation must be one of ['mean', 'max', 'sum'], got {similarity_agregation}")
    
    # Combine parameters and create optimizers/schedulers
    allParams = list(flow.parameters())
    
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    flow_loss_fn = torch.nn.MSELoss()
    decoder_1.eval()
    decoder_2.eval()

    # Get dataset sizes
    N_paired = len(trainInfo_paired)
    N_1 = len(trainInfo_1)
    N_2 = len(trainInfo_2)
    N = N_1
    if N_2>N:
        N=N_2
    all_losses = []
    all_flow_losses = []
    all_dist_losses = []
    
    for e in range(NUM_EPOCHS):        
        flow.train()
        trainloader_1 = getSamples(N_1, bs_1)
        len_1 = len(trainloader_1)
        trainloader_2 = getSamples(N_2, bs_2)
        len_2 = len(trainloader_2)
        trainloader_paired = getSamples(N_paired, bs_paired)
        len_paired = len(trainloader_paired)
        lens = [len_1,len_2,len_paired]
        maxLen = np.max(lens)
        
        if len(trainloader_1) < maxLen:
            while len(trainloader_1) < maxLen:
                trainloader_1 += getSamples(N_1, bs_1)[:maxLen - len(trainloader_1)]
        if len(trainloader_2) < maxLen:
            while len(trainloader_2) < maxLen:
                trainloader_2 += getSamples(N_2, bs_2)[:maxLen - len(trainloader_2)]
        if len(trainloader_paired) < maxLen:
            while len(trainloader_paired) < maxLen:
                trainloader_paired += getSamples(N_paired, bs_paired)[:maxLen - len(trainloader_paired)]
        
        batch_losses = []
        batch_flow_losses = []
        batch_dist_losses = []
        
        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]
            dataIndex_paired = trainloader_paired[j]

            df_pairs = trainInfo_paired.iloc[dataIndex_paired,:]
            df_1 = trainInfo_1.iloc[dataIndex_1,:]
            df_2 = trainInfo_2.iloc[dataIndex_2,:]
            
            z_1 = torch.tensor(np.concatenate((allembs1.loc[df_pairs['sig_id.x']].values,
                                                 allembs1.loc[df_1.sig_id].values))).double().to(device)
            z_2 = torch.tensor(np.concatenate((allembs2.loc[df_pairs['sig_id.y']].values,
                                           allembs2.loc[df_2.sig_id].values))).double().to(device)
            X_1 = torch.tensor(np.concatenate((allX.loc[df_pairs['sig_id.x']].values,
                                                 allX.loc[df_1.sig_id].values))).double().to(device)
            X_2 = torch.tensor(np.concatenate((allX.loc[df_pairs['sig_id.y']].values,
                                           allX.loc[df_2.sig_id].values))).double().to(device)
            
            X_2_aligned, X_1_aligned, _, _ = transact_align_gpu(
                X_2, X_1,
                n_src_pcs=75, n_tgt_pcs=75, n_pv=30,
                kernel='rbf', gamma=5e-4
            )
            X_1_aligned = X_1_aligned - X_1_aligned.mean(0)
            X_2_aligned = X_2_aligned - X_2_aligned.mean(0)
            C_pre = X_1_aligned @ X_2_aligned.T
            C_pre = C_pre / C_pre.max()
            C_pre = torch.nn.functional.relu(C_pre)
            
            cond1 = np.concatenate((df_pairs.conditionId.values, df_1.conditionId.values))
            cond2 = np.concatenate((df_pairs.conditionId.values, df_2.conditionId.values))
            C = (cond1[:, None] == cond2[None, :]).astype(np.float64)
            C = torch.tensor(C, dtype=torch.double, device=device).detach()

            # Combine C and C_pre using the specified aggregation method
            if similarity_agregation == 'mean':
                C = (C + C_pre) / 2
            elif similarity_agregation == 'max':
                C = torch.maximum(C, C_pre)
            elif similarity_agregation == 'sum':
                C = C + C_pre
            C = C.detach()

            optimizer.zero_grad()
            if tanslation_direction == '1 to 2':
                if z_1.shape[0] == z_2.shape[0]:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_1 + t * z_2
                    dz_t = z_2 - z_1
                elif z_1.shape[0] > z_2.shape[0]:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_1[:z_2.shape[0]] + t * z_2
                    dz_t = z_2 - z_1[:z_2.shape[0]]
                else:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_1 + t * z_2[:z_1.shape[0]]
                    dz_t = z_2[:z_1.shape[0]] - z_1
                
                flow_loss = flow_loss_fn(flow(z_t, t), dz_t)
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                z12 = z_1.clone()
                for step in range(n_steps):
                    z12 = flow.step(z12, time_steps[step], time_steps[step + 1])
                dist = torch.cdist(z12, z_2) * C
                dist = torch.sum(dist)
            else:
                if z_1.shape[0] == z_2.shape[0]:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_2 + t * z_1
                    dz_t = z_1 - z_2
                elif z_2.shape[0] > z_1.shape[0]:
                    t = torch.rand(len(z_1), 1).double().to(device)
                    z_t = (1 - t) * z_2[:z_1.shape[0]] + t * z_1
                    dz_t = z_1 - z_2[:z_1.shape[0]]
                else:
                    t = torch.rand(len(z_2), 1).double().to(device)
                    z_t = (1 - t) * z_2 + t * z_1[:z_2.shape[0]]
                    dz_t = z_1[:z_2.shape[0]] - z_2
                
                flow_loss = flow_loss_fn(flow(z_t, t), dz_t)
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                z21 = z_2.clone()
                for step in range(n_steps):
                    z21 = flow.step(z21, time_steps[step], time_steps[step + 1])
                dist = torch.cdist(z_1, z21) * C
                dist = torch.sum(dist)

            loss = model_params['flow_lambda'] * flow_loss + model_params['conditional_flow_lambda']*dist
            loss.backward(retain_graph=True)
            optimizer.step()

            batch_losses.append(loss.item())
            batch_flow_losses.append(flow_loss.item())
            batch_dist_losses.append(dist.item())

        scheduler.step()
        outString = 'Epoch={:.0f}/{:.0f}'.format(e+1,NUM_EPOCHS)
        outString += ', distance loss={:.4f}'.format(np.nanmean(batch_dist_losses))
        outString += ', flow loss={:.4f}'.format(np.nanmean(batch_flow_losses))
        outString += ', loss={:.4f}'.format(np.nanmean(batch_losses))

        all_losses.append(np.nanmean(batch_losses))
        all_flow_losses.append(np.nanmean(batch_flow_losses))
        all_dist_losses.append(np.nanmean(batch_dist_losses))

        if (e % 250 == 0) or (e + 1 == NUM_EPOCHS):
            print2log(outString)
    
    flow.eval()
    with torch.no_grad():
        if pairs_train is not None:
            x_1_equivalent = x_1_train[pairs_train,:].double()
            x_2_equivalent = x_2_train[pairs_train,:].double()
            z_latent_base_1_equivalent = z_1_train[pairs_train,:].double()
            z_latent_base_2_equivalent = z_2_train[pairs_train,:].double()
            cosineLoss = torch.nn.functional.cosine_similarity(z_latent_base_1_equivalent,z_latent_base_2_equivalent,dim=-1).mean()
            
            if tanslation_direction == '1 to 2':
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                for step in range(n_steps):
                    z_latent_base_1_equivalent = flow.step(z_latent_base_1_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_2_equivalent = decoder_2(z_latent_base_1_equivalent).detach()
                pearson_translation = pearson_r(x_hat_2_equivalent.flatten(), x_2_equivalent.flatten()).detach().cpu().numpy()
            else:
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.double)
                for step in range(n_steps):
                    z_latent_base_2_equivalent = flow.step(z_latent_base_2_equivalent, time_steps[step], time_steps[step + 1])
                x_hat_1_equivalent = decoder_1(z_latent_base_2_equivalent).detach()
                pearson_translation = pearson_r(x_hat_1_equivalent.flatten(), x_1_equivalent.flatten()).detach().cpu().numpy()

            print2log('cosine of trained paired conditions: {:.4f}'.format(cosineLoss.item()))
            cosineLoss = cosineLoss.item()
        else:
            pearson_translation = np.nan
            cosineLoss = np.nan

    return (pearson_translation, cosineLoss, flow)


def train_RNAseq_AE_fold(model_params, device, x_train,
                    decoder, encoder,
                    bs:int, NUM_EPOCHS:int,
                    evaluate:bool=True,
                    plot_label: str = ''):
    import os
    from matplotlib import pyplot as plt

    # Combine parameters and create optimizers/schedulers
    allParams = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    recon_criterion = NBLoss()
    # Get dataset sizes
    N = x_train.shape[0]

    # Per-epoch trackers (mean and std across batches within each epoch)
    epoch_loss_mean        = []
    epoch_loss_std         = []
    epoch_pearson_mu_mean  = []
    epoch_pearson_mu_std   = []
    epoch_pearson_var_mean = []
    epoch_pearson_var_std  = []

    # Training loop
    for e in range(NUM_EPOCHS):
        decoder.train()
        encoder.train()
        # Generate training batches
        trainloader = getSamples(N, bs)
        batch_pearson_mus = []
        batch_pearson_vars = []
        batch_mu_r2s = []
        batch_var_r2s = []
        batch_losses = []
        # Iterate through batches
        for j in range(len(trainloader)):
            dataIndex = trainloader[j]
            X = x_train[dataIndex, :].double().to(device)
            optimizer.zero_grad()
            # encode
            z = encoder(X)
            # decode
            y_pred_means, y_pred_vars = decoder(z)
            # calculate loss
            fitLoss = recon_criterion(y_pred_means, X, y_pred_vars)
            L2Loss = _l2_safe(decoder, model_params['dec_l2_reg']) + _l2_safe(encoder, model_params['enc_l2_reg'])
            loss = fitLoss + L2Loss
            loss.backward()
            optimizer.step()
            # Get performance metrics
            counts, logits = _convert_mean_disp_to_counts_logits(
                        torch.clamp(
                            y_pred_means.detach(),
                            min=1e-4,
                            max=1e4,
                        ),
                        torch.clamp(
                            y_pred_vars.detach(),
                            min=1e-4,
                            max=1e4,
                        )
                    )
            distr = NegativeBinomial(total_count=counts,
                                     logits=logits)
            nb_sample = distr.sample().cpu().numpy()
            yp_m = nb_sample.mean(0)
            yp_v = nb_sample.var(0)
            # true means and variances
            yt_m = X.detach().cpu().numpy().mean(axis=0)
            yt_v = X.detach().cpu().numpy().var(axis=0)
            pearson_mu,_ = pearsonr(yp_m, yt_m)
            pearson_var,_ = pearsonr(yp_v, yt_v)

            batch_pearson_mus.append(pearson_mu)
            batch_pearson_vars.append(pearson_var)
            batch_mu_r2s.append(r2_score(yt_m, yp_m))
            batch_var_r2s.append(r2_score(yt_v, yp_v))
            batch_losses.append(loss.item())

        scheduler.step()
        epoch_loss_mean.append(np.nanmean(batch_losses))
        epoch_loss_std.append(np.nanstd(batch_losses))
        epoch_pearson_mu_mean.append(np.nanmean(batch_pearson_mus))
        epoch_pearson_mu_std.append(np.nanstd(batch_pearson_mus))
        epoch_pearson_var_mean.append(np.nanmean(batch_pearson_vars))
        epoch_pearson_var_std.append(np.nanstd(batch_pearson_vars))

        outString = 'Epoch={:.0f}/{:.0f}'.format(e+1,NUM_EPOCHS)
        outString += ', muR2={:.4f}'.format(np.nanmean(batch_mu_r2s))
        outString += ', varR2={:.4f}'.format(np.nanmean(batch_var_r2s))
        outString += ', pearson_mu={:.4f}'.format(epoch_pearson_mu_mean[-1])
        outString += ', pearson_var={:.4f}'.format(epoch_pearson_var_mean[-1])
        outString += ', loss={:.4f}'.format(epoch_loss_mean[-1])

        # Logging
        if (e % 5 == 0) or (e + 1 == NUM_EPOCHS):
            print2log(outString)

    # Save training curves figure
    os.makedirs('training_plots', exist_ok=True)
    fname_tag = f'_{plot_label}' if plot_label else ''
    epochs_arr = np.arange(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    def _fill_ae(ax, mean, std, title, ylabel, color):
        mean, std = np.array(mean), np.array(std)
        ax.plot(epochs_arr, mean, color=color)
        ax.fill_between(epochs_arr, mean - std, mean + std, alpha=0.3, color=color)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)

    _fill_ae(axes[0], epoch_loss_mean,        epoch_loss_std,        'Loss',             'Loss', 'steelblue')
    _fill_ae(axes[1], epoch_pearson_mu_mean,  epoch_pearson_mu_std,  'Pearson r (mean)', 'r',    'darkorange')
    _fill_ae(axes[2], epoch_pearson_var_mean, epoch_pearson_var_std, 'Pearson r (var)',  'r',    'seagreen')
    plt.suptitle('AE training (NB)', fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join('training_plots', f'ae_nb{fname_tag}.png')
    fig.savefig(fig_path, dpi=150); plt.close(fig)
    print2log(f'Training plot saved to {os.path.abspath(fig_path)}')

        
    # evaluate
    if evaluate:
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            y_pred_mu, y_pred_var = decoder(encoder(x_train.double().to(device)))
            # Get performance metrics
            counts, logits = _convert_mean_disp_to_counts_logits(
                        torch.clamp(
                            y_pred_means.detach(),
                            min=1e-4,
                            max=1e4,
                        ),
                        torch.clamp(
                            y_pred_vars.detach(),
                            min=1e-4,
                            max=1e4,
                        )
                    )
            distr = NegativeBinomial(total_count=counts,
                                     logits=logits)
            nb_sample = distr.sample().cpu().numpy()
            yp_m = nb_sample.mean(0)
            yp_v = nb_sample.var(0)
            # true means and variances
            yt_m = X.detach().cpu().numpy().mean(axis=0)
            yt_v = X.detach().cpu().numpy().var(axis=0)
            pearson_mu,_ = pearsonr(yp_m, yt_m)
            pearson_var,_ = pearsonr(yp_v, yt_v)
            r2_mu = r2_score(yt_m, yp_m)
            r2_var = r2_score(yt_v, yp_v)
    else:
        pearson_mu = None
        pearson_var = None
        r2_mu = None
        r2_var = None
    metrics = {
        'pearson_mu': pearson_mu,
        'pearson_var': pearson_var,
        'r2_mu': r2_mu,
        'r2_var': r2_var
    }
    return (metrics, decoder, encoder)

def train_RNAseq_AE_fold_gauss(model_params, device, x_train,
                               decoder, encoder,
                               bs:int, NUM_EPOCHS:int,
                               evaluate:bool=True,
                               plot_label: str = ''):
    """Same as train_RNAseq_AE_fold but with Gaussian NLL — for log1p+quantile-normalized data."""
    import os
    from matplotlib import pyplot as plt

    allParams = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    recon_criterion = torch.nn.GaussianNLLLoss(reduction='mean', eps=1e-6)
    N = x_train.shape[0]

    # Per-epoch trackers (mean and std across batches within each epoch)
    epoch_loss_mean        = []
    epoch_loss_std         = []
    epoch_pearson_mu_mean  = []
    epoch_pearson_mu_std   = []
    epoch_pearson_var_mean = []
    epoch_pearson_var_std  = []

    for e in range(NUM_EPOCHS):
        decoder.train(); encoder.train()
        trainloader = getSamples(N, bs)
        batch_pearson_mus, batch_pearson_vars = [], []
        batch_mu_r2s, batch_var_r2s, batch_losses = [], [], []
        for j in range(len(trainloader)):
            dataIndex = trainloader[j]
            X = x_train[dataIndex, :].float().to(device)        # was .double()
            optimizer.zero_grad()
            z = encoder(X)
            y_pred_means, y_pred_vars = decoder(z)
            fitLoss = recon_criterion(y_pred_means, X, y_pred_vars)
            L2Loss = (_l2_safe(decoder, model_params['dec_l2_reg'])
                      + _l2_safe(encoder, model_params['enc_l2_reg']))
            loss = fitLoss + L2Loss
            loss.backward()
            optimizer.step()
            # No sampling — use predicted moments directly.
            yp_m = y_pred_means.detach().cpu().numpy().mean(axis=0)
            yp_v = y_pred_vars.detach().cpu().numpy().mean(axis=0)
            yt_m = X.detach().cpu().numpy().mean(axis=0)
            yt_v = X.detach().cpu().numpy().var(axis=0)
            pm, _ = pearsonr(yp_m, yt_m)
            pv, _ = pearsonr(yp_v, yt_v)
            batch_pearson_mus.append(pm); batch_pearson_vars.append(pv)
            batch_mu_r2s.append(r2_score(yt_m, yp_m))
            batch_var_r2s.append(r2_score(yt_v, yp_v))
            batch_losses.append(loss.item())
        scheduler.step()
        epoch_loss_mean.append(np.nanmean(batch_losses))
        epoch_loss_std.append(np.nanstd(batch_losses))
        epoch_pearson_mu_mean.append(np.nanmean(batch_pearson_mus))
        epoch_pearson_mu_std.append(np.nanstd(batch_pearson_mus))
        epoch_pearson_var_mean.append(np.nanmean(batch_pearson_vars))
        epoch_pearson_var_std.append(np.nanstd(batch_pearson_vars))
        if (e % 5 == 0) or (e + 1 == NUM_EPOCHS):
            print2log('Epoch={:.0f}/{:.0f}, muR2={:.4f}, varR2={:.4f}, '
                      'pearson_mu={:.4f}, pearson_var={:.4f}, loss={:.4f}'.format(
                          e+1, NUM_EPOCHS,
                          np.nanmean(batch_mu_r2s), np.nanmean(batch_var_r2s),
                          epoch_pearson_mu_mean[-1], epoch_pearson_var_mean[-1],
                          epoch_loss_mean[-1]))

    # Save training curves figure
    os.makedirs('training_plots', exist_ok=True)
    fname_tag = f'_{plot_label}' if plot_label else ''
    epochs = np.arange(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    def _fill_ae(ax, mean, std, title, ylabel, color):
        mean, std = np.array(mean), np.array(std)
        ax.plot(epochs, mean, color=color)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.3, color=color)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)

    _fill_ae(axes[0], epoch_loss_mean,        epoch_loss_std,        'Loss',               'Loss', 'steelblue')
    _fill_ae(axes[1], epoch_pearson_mu_mean,  epoch_pearson_mu_std,  'Pearson r (mean)',   'r',    'darkorange')
    _fill_ae(axes[2], epoch_pearson_var_mean, epoch_pearson_var_std, 'Pearson r (var)',    'r',    'seagreen')
    plt.suptitle('AE training (Gaussian)', fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join('training_plots', f'ae_gauss{fname_tag}.png')
    fig.savefig(fig_path, dpi=150); plt.close(fig)
    print2log(f'Training plot saved to {os.path.abspath(fig_path)}')

    pearson_mu = pearson_var = r2_mu = r2_var = None
    if evaluate:
        encoder.eval(); decoder.eval()
        # batched eval (x_train can be huge — don't push it whole to GPU)
        BS_EVAL = 4096
        sums_mu = sums_var = 0
        n = 0
        agg_yp_m = np.zeros(x_train.shape[1], dtype=np.float64)
        agg_yp_v = np.zeros(x_train.shape[1], dtype=np.float64)
        agg_yt_m = np.zeros(x_train.shape[1], dtype=np.float64)
        agg_yt_v = np.zeros(x_train.shape[1], dtype=np.float64)
        with torch.no_grad():
            for c0 in range(0, N, BS_EVAL):
                c1 = min(c0 + BS_EVAL, N)
                idx = np.arange(c0, c1)
                X = x_train[idx, :].float().to(device)
                y_mu, y_var = decoder(encoder(X))
                k = c1 - c0
                agg_yp_m += y_mu.cpu().numpy().sum(axis=0); agg_yp_v += y_var.cpu().numpy().sum(axis=0)
                agg_yt_m += X.cpu().numpy().sum(axis=0);    agg_yt_v += (X.cpu().numpy()**2).sum(axis=0)
                n += k
        yp_m = agg_yp_m / n; yp_v = agg_yp_v / n
        yt_m = agg_yt_m / n; yt_v = agg_yt_v / n - yt_m**2
        pearson_mu, _  = pearsonr(yp_m, yt_m)
        pearson_var, _ = pearsonr(yp_v, yt_v)
        r2_mu  = r2_score(yt_m, yp_m)
        r2_var = r2_score(yt_v, yp_v)

    metrics = {'pearson_mu': pearson_mu, 'pearson_var': pearson_var,
               'r2_mu': r2_mu, 'r2_var': r2_var}
    return (metrics, decoder, encoder)


def train_RNAseq_flowMatch_fold(model_params, device, x_1_train, x_2_train,z_1_train,z_2_train,flow,
                    bs_1:int, bs_2:int, NUM_EPOCHS:int,
                    translation_direction = '1 to 2',
                    plot_label: str = ''):
    import os
    from matplotlib import pyplot as plt

    # Combine parameters and create optimizers/schedulers
    allParams = list(flow.parameters())
    optimizer = torch.optim.Adam(allParams, lr=model_params['encoding_lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=model_params['schedule_step_enc'],
                                                gamma=model_params['gamma_enc'])
    flow_loss_fn = torch.nn.MSELoss()

    # Get dataset sizes
    N_1 = x_1_train.shape[0]
    N_2 = x_2_train.shape[0]

    # Per-epoch trackers (mean and std across batches)
    epoch_loss_mean      = []
    epoch_loss_std       = []
    epoch_flow_loss_mean = []
    epoch_flow_loss_std  = []
    epoch_dist_loss_mean = []
    epoch_dist_loss_std  = []

    # Training loop
    for e in range(NUM_EPOCHS):
        flow.train()
        # Generate training batches
        trainloader_1 = getSamples(N_1, bs_1)
        trainloader_2 = getSamples(N_2, bs_2)
        maxLen = max(len(trainloader_1), len(trainloader_2))
        # Pad shorter trainloader if needed
        if len(trainloader_1) < maxLen:
            while len(trainloader_1) < maxLen:
                trainloader_1 += getSamples(N_1, bs_1)[:maxLen - len(trainloader_1)]
        if len(trainloader_2) < maxLen:
            while len(trainloader_2) < maxLen:
                trainloader_2 += getSamples(N_2, bs_2)[:maxLen - len(trainloader_2)]
        # Iterate through batches
        batch_losses = []
        batch_flow_losses = []
        batch_dist_losses = []
        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]
            # Get data for the batch
            X_1 = x_1_train[dataIndex_1, :]#.cpu().numpy()#.double().to(device)
            X_2 = x_2_train[dataIndex_2, :]#.cpu().numpy()#.double().to(device)
            z_1 = z_1_train[dataIndex_1, :]#.double().to(device)
            z_2 = z_2_train[dataIndex_2, :]#.double().to(device)
            X_2_aligned, X_1_aligned, _,_ = transact_align_gpu(
                X_2,        # source → will become Z_source
                X_1,        # target → will become Z_target
                n_src_pcs=75,
                n_tgt_pcs=75,
                n_pv=30,
                kernel='rbf',
                gamma=5e-4      # or whatever you tuned
            )
            X_1_aligned = X_1_aligned  - X_1_aligned.mean(0)
            X_2_aligned = X_2_aligned  - X_2_aligned.mean(0)
            C = X_1_aligned @ X_2_aligned.T
            # Using z_1 and z_2 as the latent representations for flow matching
            # Then in each iteration:
            C = C/C.max()
            optimizer.zero_grad()
            if translation_direction == '1 to 2':
                # Translate z1 to z2 with flow
                if z_1.shape[0] == z_2.shape[0]:
                    t = torch.rand(len(z_1), 1).float().to(device)
                    z_t = (1 - t) * z_1 + t * z_2
                    dz_t = z_2 - z_1
                elif z_1.shape[0] > z_2.shape[0]:
                    t = torch.rand(len(z_2), 1).float().to(device)
                    z_t = (1 - t) * z_1[:z_2.shape[0]] + t * z_2
                    dz_t = z_2 - z_1[:z_2.shape[0]]
                else:
                    t = torch.rand(len(z_1), 1).float().to(device)
                    z_t = (1 - t) * z_1 + t * z_2[:z_1.shape[0]]
                    dz_t = z_2[:z_1.shape[0]] - z_1
                flow_loss = flow_loss_fn(flow(z_t, t), dz_t)
                ## Conditioned based on initial correlation
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.float)
                z12 = z_1.clone()
                for step in range(n_steps):
                    z12 = flow.step(z12, time_steps[step], time_steps[step + 1])
                dist = torch.cdist(z12, z_2) * torch.nn.functional.relu(C)
                dist = torch.sum(dist)
            else:
                # Translate now z2 to z1
                if z_1.shape[0] == z_2.shape[0]:
                    t = torch.rand(len(z_2), 1).float().to(device)
                    z_t = (1 - t) * z_2 + t * z_1
                    dz_t = z_1 - z_2
                elif z_2.shape[0] > z_1.shape[0]:
                    t = torch.rand(len(z_1), 1).float().to(device)
                    z_t = (1 - t) * z_2[:z_1.shape[0]] + t * z_1
                    dz_t = z_1 - z_2[:z_1.shape[0]]
                else:
                    t = torch.rand(len(z_2), 1).float().to(device)
                    z_t = (1 - t) * z_2 + t * z_1[:z_2.shape[0]]
                    dz_t = z_1[:z_2.shape[0]] - z_2
                flow_loss = flow_loss_fn(flow(z_t, t), dz_t)
                ## Conditioned based on initial correlation
                n_steps = 10
                time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.float)
                z21 = z_2.clone()
                for step in range(n_steps):
                    z21 = flow.step(z21, time_steps[step], time_steps[step + 1])
                dist = torch.cdist(z_1, z21) * torch.nn.functional.relu(C)
                dist = torch.sum(dist)
            loss = model_params['flow_lambda'] * flow_loss + model_params['conditional_flow_lambda']*dist
            loss.backward(retain_graph=True)
            optimizer.step()

            batch_losses.append(loss.item())
            batch_flow_losses.append(flow_loss.item())
            batch_dist_losses.append(dist.item())

        scheduler.step()
        epoch_loss_mean.append(np.nanmean(batch_losses))
        epoch_loss_std.append(np.nanstd(batch_losses))
        epoch_flow_loss_mean.append(np.nanmean(batch_flow_losses))
        epoch_flow_loss_std.append(np.nanstd(batch_flow_losses))
        epoch_dist_loss_mean.append(np.nanmean(batch_dist_losses))
        epoch_dist_loss_std.append(np.nanstd(batch_dist_losses))

        outString = 'Epoch={:.0f}/{:.0f}'.format(e+1,NUM_EPOCHS)
        outString += ', distance loss={:.4f}'.format(epoch_dist_loss_mean[-1])
        outString += ', flow loss={:.4f}'.format(epoch_flow_loss_mean[-1])
        outString += ', loss={:.4f}'.format(epoch_loss_mean[-1])
        print2log(outString)

    # Save training curves figure
    os.makedirs('training_plots', exist_ok=True)
    safe_dir = translation_direction.replace(' ', '_')
    fname_tag = f'_{plot_label}' if plot_label else ''
    epochs = np.arange(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    def _fill(ax, mean, std, title, color):
        mean, std = np.array(mean), np.array(std)
        ax.plot(epochs, mean, color=color)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.3, color=color)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')

    _fill(axes[0], epoch_loss_mean,      epoch_loss_std,      'Total Loss',    'steelblue')
    _fill(axes[1], epoch_flow_loss_mean, epoch_flow_loss_std, 'Flow Loss',     'darkorange')
    _fill(axes[2], epoch_dist_loss_mean, epoch_dist_loss_std, 'Distance Loss', 'seagreen')
    plt.suptitle(f'Flow training — direction: {translation_direction}', fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join('training_plots', f'flow_{safe_dir}{fname_tag}.png')
    fig.savefig(fig_path, dpi=150); plt.close(fig)
    print2log(f'Training plot saved to {os.path.abspath(fig_path)}')

    flow.eval()
    with torch.no_grad():
        if translation_direction == '1 to 2':
            n_steps = 10
            time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.float)
            ztran = z_1_train.clone()
            for step in range(n_steps):
                ztran = flow.step(ztran, time_steps[step], time_steps[step + 1])
        else:
            n_steps = 10
            time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.float)
            ztran = z_2_train.clone()
            for step in range(n_steps):
                ztran = flow.step(ztran, time_steps[step], time_steps[step + 1])
    return (ztran,flow)


def validate_RNAseq_flowMatch_fold(device, x_1_test, x_2_test,
                                   encoder_1, encoder_2, flow,
                                   translation_direction='1 to 2',
                                   bs=4096):
    encoder_1.eval(); encoder_2.eval(); flow.eval()
    def _encode(enc, X):
        N = X.shape[0]; out = []
        for c0 in range(0, N, bs):
            c1 = min(c0 + bs, N)
            xb = X[np.arange(c0, c1), :].to(device)   # works on LazyMatrix or tensor
            out.append(enc(xb).detach())
        return torch.cat(out, dim=0)

    with torch.no_grad():
        z1 = _encode(encoder_1, x_1_test)
        z2 = _encode(encoder_2, x_2_test)
        n_steps = 10
        time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.float)
        ztran = (z1 if translation_direction == '1 to 2' else z2).clone()
        for step in range(n_steps):
            ztran = flow.step(ztran, time_steps[step], time_steps[step + 1])
    return ztran