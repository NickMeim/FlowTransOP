#!/usr/bin/env python3
"""
Training script for a single CV fold.
Submit as array job: sbatch --array=0-9 train_job.sh
"""
import argparse
from pathlib import Path
import torch
import numpy as np
from models import VarDecoder,SimpleEncoder, Flow, ElementWiseLinear
from trainingUtils import train_RNAseq_AE_fold_gauss, train_RNAseq_flowMatch_fold, validate_RNAseq_flowMatch_fold
from utility import *
from transact_utility_gpu import *
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import logging
from logging import FileHandler
import warnings
warnings.filterwarnings('ignore', message='.*ks_2samp.*')


# Configuration
DATA_DIR    = Path("../archs4")
PREPROC_DIR = DATA_DIR / "preprocessed"
MODEL_DIR   = Path("../archs4/models")


# --- LazyMatrix: sample-major memmap with optional row reindex --
class LazyMatrix:
    """Behaves like a 2D CPU tensor for x[idx, :] — returns torch.float32 on the fly."""
    def __init__(self, mat_path: Path, row_index: np.ndarray = None):
        self._mat = np.load(mat_path, mmap_mode='r')   # (N_total, G), float32
        self._row_index = row_index                     # logical -> physical row
    @property
    def shape(self):
        n = self._row_index.shape[0] if self._row_index is not None else self._mat.shape[0]
        return (n, self._mat.shape[1])
    def __len__(self):
        return self.shape[0]
    def __getitem__(self, key):
        row_key, col_key = (key if isinstance(key, tuple) else (key, slice(None)))
        phys = self._row_index[row_key] if self._row_index is not None else row_key
        block = np.asarray(self._mat[phys])
        if not (isinstance(col_key, slice) and col_key == slice(None)):
            block = block[:, col_key]
        return torch.from_numpy(np.ascontiguousarray(block)).float()


def load_fold(species: str, fold: int):
    train_i = np.load(PREPROC_DIR / f"{species}_fold{fold}_train_idx.npy")
    val_i   = np.load(PREPROC_DIR / f"{species}_fold{fold}_val_idx.npy")
    mat     = PREPROC_DIR / f"{species}_X.npy"
    return LazyMatrix(mat, row_index=train_i), LazyMatrix(mat, row_index=val_i)

class PermutedLazy:
    def __init__(self, base, col_perm): self.base, self.cp = base, col_perm
    @property
    def shape(self): return self.base.shape
    def __len__(self): return len(self.base)
    def __getitem__(self, key):
        t = self.base[key]
        return t[:, torch.as_tensor(self.cp, dtype=torch.long)]
    
@torch.no_grad()
def encode_all(encoder, X_lazy, device, bs=4096):
    encoder.eval()
    N = len(X_lazy)
    out = []
    for c0 in range(0, N, bs):
        c1 = min(c0 + bs, N)
        X = X_lazy[np.arange(c0, c1), :].to(device)
        out.append(encoder(X).cpu())
    return torch.cat(out, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for traiming.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--enc_l2_reg', type=float, default=0.001, help='L2 regularization for the encoder.')
    parser.add_argument('--dec_l2_reg', type=float, default=0.001, help='L2 regularization for the decoder.')
    parser.add_argument('--encoding_lr', type=float, default=0.001, help='Learning rate for the encoder.')
    parser.add_argument('--schedule_step_enc', type=int, default=20, help='Step size for the encoder learning rate scheduler.')
    parser.add_argument('--gamma_enc', type=float, default=0.8, help='Gamma for the encoder learning rate scheduler.')
    parser.add_argument('--autoencoder_wd', type=float, default=0.0, help='Weight decay for the autoencoder.')
    # Model parameters
    parser.add_argument('--encoder_1_hiddens', type=int, nargs='+', default=[4096, 2048, 1024, 512], help='Hidden layer sizes for encoder 1.')
    parser.add_argument('--encoder_2_hiddens', type=int, nargs='+', default=[4096, 2048, 1024, 512], help='Hidden layer sizes for encoder 2.')
    parser.add_argument('--latent_dim', type=int, default=512, help='Dimension of the latent space.')
    parser.add_argument('--decoder_1_hiddens', type=int, nargs='+', default=[512, 1024, 2048, 4096], help='Hidden layer sizes for decoder 1.')
    parser.add_argument('--decoder_2_hiddens', type=int, nargs='+', default=[512, 1024, 2048, 4096], help='Hidden layer sizes for decoder 2.')
    parser.add_argument('--dropout_decoder', type=float, default=0.2, help='Dropout rate for the decoder.')
    parser.add_argument('--dropout_encoder', type=float, default=0.2, help='Dropout rate for the encoder.')
    parser.add_argument('--bn_decoder', type=float, default=0.6, help='Use batch normalization in the decoder.')
    parser.add_argument('--bn_encoder', type=float, default=0.6, help='Use batch normalization in the encoder.')
    parser.add_argument('--dropout_input_encoder', type=float, default=0.5, help='Dropout rate for the imput of the encoder.')
    parser.add_argument('--dropout_input_decoder', type=float, default=0.2, help='Dropout rate for the imput of the decoder.')
    parser.add_argument('--encoder_activation', type=str, 
                        choices=['LeakyReLU', 'ReLU', 'ELU', 'Sigmoid'],  
                        help='Activation function used between layers of the encoder',
                        default='ELU')
    parser.add_argument('--decoder_activation', type=str,
                        choices=['LeakyReLU', 'ReLU', 'ELU', 'Sigmoid'],  
                        help='Activation function used between layers of the decoder',
                        default='ELU')
    ## arguments for stretching and alinging
    parser.add_argument('--flow_lambda', type=float, default=1., help='Flow matrix regularization parameter.')
    parser.add_argument('--conditional_flow_lambda', type=float, default=1e-3, help='Flow matching regularization parameter.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_file ='logs/ARCHS4_fold_' + str(args.fold) + '.log'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = FileHandler(log_file, mode='a')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    print2log = logger.info
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print2log(f"Using device: {device}")

    if args.decoder_activation == 'LeakyReLU':
        decoder_activation = torch.nn.LeakyReLU(0.01)
    elif args.decoder_activation == 'ReLU':
        decoder_activation = torch.nn.ReLU()
    elif args.decoder_activation == 'ELU':
        decoder_activation = torch.nn.ELU()
    elif args.decoder_activation == 'Sigmoid':
        decoder_activation = torch.nn.Sigmoid()

    if args.encoder_activation == 'LeakyReLU':
        encoder_activation = torch.nn.LeakyReLU(0.01)
    elif args.encoder_activation == 'ReLU':
        encoder_activation = torch.nn.ReLU()
    elif args.encoder_activation == 'ELU':
        encoder_activation = torch.nn.ELU()
    elif args.encoder_activation == 'Sigmoid':
        encoder_activation = torch.nn.Sigmoid()

    # Model parameters
    model_params = {
        'encoder_1_hiddens': args.encoder_1_hiddens,
        'encoder_2_hiddens': args.encoder_2_hiddens,
        'latent_dim': args.latent_dim,
        'decoder_1_hiddens': args.decoder_1_hiddens,
        'decoder_2_hiddens': args.decoder_2_hiddens,
        'dropout_decoder': args.dropout_decoder,
        'dropout_encoder': args.dropout_encoder,
        'encoder_activation': encoder_activation,
        'decoder_activation': decoder_activation,
        'bn_encoder': args.bn_encoder,
        'bn_decoder': args.bn_decoder,
        'dropout_input_encoder': args.dropout_input_encoder,
        'dropout_input_decoder': args.dropout_input_decoder,
        'encoding_lr': args.encoding_lr,
        'schedule_step_enc': args.schedule_step_enc,
        'gamma_enc': args.gamma_enc,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'enc_l2_reg': args.enc_l2_reg,
        'dec_l2_reg': args.dec_l2_reg,
        'autoencoder_wd': args.autoencoder_wd,
        'flow_lambda': args.flow_lambda,
        'conditional_flow_lambda': args.conditional_flow_lambda
    }
    
    # Load preprocessed fold data (mmaps, lazy)
    print2log(f"Loading preprocessed fold {args.fold} ...")
    X_human,  X_human_val  = load_fold("human", args.fold)
    X_mouse,  X_mouse_val  = load_fold("mouse", args.fold)
    print2log(f"Human train: {X_human.shape}, val: {X_human_val.shape}")
    print2log(f"Mouse train: {X_mouse.shape}, val: {X_mouse_val.shape}")

    # === Models — note loss='gauss' on every VarDecoder ===
    encoder_human = torch.nn.Sequential(ElementWiseLinear(X_human.shape[1]),
        SimpleEncoder(X_human.shape[1], model_params['encoder_1_hiddens'], model_params['latent_dim'],
                    dropRate=model_params['dropout_encoder'], bn=model_params['bn_encoder'],
                    activation=model_params['encoder_activation'], dropIn=model_params['dropout_input_encoder'],
                    dtype=torch.float)).to(device)
    encoder_mouse = torch.nn.Sequential(ElementWiseLinear(X_mouse.shape[1]),
        SimpleEncoder(X_mouse.shape[1], model_params['encoder_2_hiddens'], model_params['latent_dim'],
                    dropRate=model_params['dropout_encoder'], bn=model_params['bn_encoder'],
                    activation=model_params['encoder_activation'], dropIn=model_params['dropout_input_encoder'],
                    dtype=torch.float)).to(device)
    decoder_human = VarDecoder(model_params['latent_dim'], model_params['decoder_1_hiddens'], X_human.shape[1],
                            dropRate=model_params['dropout_decoder'], bn=model_params['bn_decoder'],
                            activation=model_params['decoder_activation'], dropIn=model_params['dropout_input_decoder'],
                            loss='gauss', dtype=torch.float).to(device)            # <- gauss
    decoder_mouse = VarDecoder(model_params['latent_dim'], model_params['decoder_2_hiddens'], X_mouse.shape[1],
                            dropRate=model_params['dropout_decoder'], bn=model_params['bn_decoder'],
                            activation=model_params['decoder_activation'], dropIn=model_params['dropout_input_decoder'],
                            loss='gauss', dtype=torch.float).to(device)            # <- gauss
    flow_h2m = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2), dtype=torch.float).to(device)

    # === AE training — Gaussian variant ===
    _, decoder_human, encoder_human = train_RNAseq_AE_fold_gauss(
        model_params, device, X_human, decoder_human, encoder_human,
        model_params['batch_size'], model_params['epochs'], evaluate=False,
        plot_label=f'human_fold{args.fold}')
    _, decoder_mouse, encoder_mouse = train_RNAseq_AE_fold_gauss(
        model_params, device, X_mouse, decoder_mouse, encoder_mouse,
        model_params['batch_size'], model_params['epochs'], evaluate=False,
        plot_label=f'mouse_fold{args.fold}')

    # === Validation — direct Gaussian, batched (no NB sampling) ===
    def gauss_recon_metrics(encoder, decoder, X_lazy, device, bs=4096):
        encoder.eval(); decoder.eval()
        N, G = X_lazy.shape
        yp_m = np.zeros(G); yp_v = np.zeros(G); yt_m = np.zeros(G); yt_sq = np.zeros(G); n = 0
        with torch.no_grad():
            for c0 in range(0, N, bs):
                c1 = min(c0 + bs, N)
                X = X_lazy[np.arange(c0, c1), :].to(device)
                mu, var = decoder(encoder(X))
                yp_m += mu.cpu().numpy().sum(axis=0)
                yp_v += var.cpu().numpy().sum(axis=0)
                yt_m += X.cpu().numpy().sum(axis=0)
                yt_sq += (X.cpu().numpy()**2).sum(axis=0)
                n += (c1 - c0)
        yp_m /= n; yp_v /= n; yt_m /= n; yt_v = yt_sq/n - yt_m**2
        pm,_ = pearsonr(yp_m, yt_m); pv,_ = pearsonr(yp_v, yt_v)
        return pm, pv, r2_score(yt_m, yp_m), r2_score(yt_v, yp_v)

    pm_h, pv_h, r2m_h, r2v_h = gauss_recon_metrics(encoder_human, decoder_human, X_human_val, device)
    pm_m, pv_m, r2m_m, r2v_m = gauss_recon_metrics(encoder_mouse, decoder_mouse, X_mouse_val, device)
    print2log(f"Validation Pearson - Human: mu={pm_h:.4f}, var={pv_h:.4f}; Mouse: mu={pm_m:.4f}, var={pv_m:.4f}")
    print2log(f"Validation R²      - Human: mu={r2m_h:.4f}, var={r2v_h:.4f}; Mouse: mu={r2m_m:.4f}, var={r2v_m:.4f}")

    # Get latent representations
    Z_human = encode_all(encoder_human, X_human, device).to(device)
    Z_mouse = encode_all(encoder_mouse, X_mouse, device).to(device)

    # Train flow models
    z_h2m, flow_h2m = train_RNAseq_flowMatch_fold(
        model_params, device,
        X_human, X_mouse,
        Z_human, Z_mouse,
        flow_h2m,
        model_params['batch_size'], model_params['batch_size'], model_params['epochs'],
        translation_direction='1 to 2',
        plot_label=f'fold{args.fold}'
    )
    
    # Save normal model
    torch.save({
        'encoder_human': encoder_human.state_dict(),
        'encoder_mouse': encoder_mouse.state_dict(),
        'decoder_human': decoder_human.state_dict(),
        'decoder_mouse': decoder_mouse.state_dict(),
        'flow_h2m': flow_h2m.state_dict(),
        'args':vars(args),
    }, MODEL_DIR / f"fold_{args.fold}_normal.pt")

    # Evaluate flow model on validation data
    flow_h2m.eval()
    with torch.no_grad():
        z_h2m_val = validate_RNAseq_flowMatch_fold(device,
                                                   X_human_val, X_mouse_val,
                                                   encoder_human, encoder_mouse,
                                                   flow_h2m,
                                                   translation_direction='1 to 2')

    # Compute val latent reps separately for saving
    Z_human_val = encode_all(encoder_human, X_human_val, device)   # CPU tensor
    Z_mouse_val = encode_all(encoder_mouse, X_mouse_val, device)

    # Save validation latent variables
    np.save(MODEL_DIR / f"fold_{args.fold}_z_h2m_val.npy",   z_h2m_val.cpu().numpy())
    np.save(MODEL_DIR / f"fold_{args.fold}_z_human_val.npy", Z_human_val.numpy())
    np.save(MODEL_DIR / f"fold_{args.fold}_z_mouse_val.npy", Z_mouse_val.numpy())
    # Save train latent variables
    np.save(MODEL_DIR / f"fold_{args.fold}_z_h2m_train.npy",   z_h2m.cpu().numpy())
    np.save(MODEL_DIR / f"fold_{args.fold}_z_human_train.npy", Z_human.cpu().numpy())
    np.save(MODEL_DIR / f"fold_{args.fold}_z_mouse_train.npy", Z_mouse.cpu().numpy())
    
    
    # ====== PERMUTED MODEL ======
    print2log("\n=== Training PERMUTED model ===")

    # Permutation indices (fixed for this fold)
    perm_idx_human = np.random.permutation(X_human.shape[1])
    perm_idx_mouse = np.random.permutation(X_mouse.shape[1])

    X_human_permuted     = PermutedLazy(X_human,     perm_idx_human)
    X_mouse_permuted     = PermutedLazy(X_mouse,     perm_idx_mouse)

    # Reinitialize models — Gaussian decoder, float dtype
    encoder_human_perm = torch.nn.Sequential(
        ElementWiseLinear(X_human.shape[1]),
        SimpleEncoder(X_human.shape[1], model_params['encoder_1_hiddens'], model_params['latent_dim'],
                    dropRate=model_params['dropout_encoder'], bn=model_params['bn_encoder'],
                    activation=model_params['encoder_activation'],
                    dropIn=model_params['dropout_input_encoder'],
                    dtype=torch.float)).to(device)
    encoder_mouse_perm = torch.nn.Sequential(
        ElementWiseLinear(X_mouse.shape[1]),
        SimpleEncoder(X_mouse.shape[1], model_params['encoder_2_hiddens'], model_params['latent_dim'],
                    dropRate=model_params['dropout_encoder'], bn=model_params['bn_encoder'],
                    activation=model_params['encoder_activation'],
                    dropIn=model_params['dropout_input_encoder'],
                    dtype=torch.float)).to(device)
    decoder_human_perm = VarDecoder(model_params['latent_dim'], model_params['decoder_1_hiddens'],
                                    X_human.shape[1],
                                    dropRate=model_params['dropout_decoder'], bn=model_params['bn_decoder'],
                                    activation=model_params['decoder_activation'],
                                    dropIn=model_params['dropout_input_decoder'],
                                    loss='gauss', dtype=torch.float).to(device)
    decoder_mouse_perm = VarDecoder(model_params['latent_dim'], model_params['decoder_2_hiddens'],
                                    X_mouse.shape[1],
                                    dropRate=model_params['dropout_decoder'], bn=model_params['bn_decoder'],
                                    activation=model_params['decoder_activation'],
                                    dropIn=model_params['dropout_input_decoder'],
                                    loss='gauss', dtype=torch.float).to(device)
    flow_h2m_perm = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2),
                        dtype=torch.float).to(device)

    # Train autoencoders on permuted features (Gaussian variant)
    _, decoder_human_perm, encoder_human_perm = train_RNAseq_AE_fold_gauss(
        model_params, device, X_human_permuted,
        decoder_human_perm, encoder_human_perm,
        model_params['batch_size'], model_params['epochs'],
        evaluate=False, plot_label=f'permuted_human_fold{args.fold}')
    _, decoder_mouse_perm, encoder_mouse_perm = train_RNAseq_AE_fold_gauss(
        model_params, device, X_mouse_permuted,
        decoder_mouse_perm, encoder_mouse_perm,
        model_params['batch_size'], model_params['epochs'],
        evaluate=False, plot_label=f'permuted_mouse_fold{args.fold}')

    # Validation reconstruction metrics — direct Gaussian, batched, no NB sampling
    pm_h_p, pv_h_p, r2m_h_p, r2v_h_p = gauss_recon_metrics(
        encoder_human_perm, decoder_human_perm, X_human_val, device)
    pm_m_p, pv_m_p, r2m_m_p, r2v_m_p = gauss_recon_metrics(
        encoder_mouse_perm, decoder_mouse_perm, X_mouse_val, device)

    print2log(f"Shuffled X Validation Pearson - Human: mu={pm_h_p:.4f}, var={pv_h_p:.4f}; "
            f"Mouse: mu={pm_m_p:.4f}, var={pv_m_p:.4f}")
    print2log(f"Shuffled X Validation R²      - Human: mu={r2m_h_p:.4f}, var={r2v_h_p:.4f}; "
            f"Mouse: mu={r2m_m_p:.4f}, var={r2v_m_p:.4f}")

    # Latent representations on permuted training data — batched
    Z_human_perm = encode_all(encoder_human_perm, X_human_permuted, device)
    Z_mouse_perm = encode_all(encoder_mouse_perm, X_mouse_permuted, device)

    # Train flow model on permuted features
    z_h2m_perm, flow_h2m_perm = train_RNAseq_flowMatch_fold(
        model_params, device,
        X_human_permuted, X_mouse_permuted,
        Z_human_perm.to(device), Z_mouse_perm.to(device),
        flow_h2m_perm,
        model_params['batch_size'], model_params['batch_size'], model_params['epochs'],
        translation_direction='1 to 2', plot_label=f'permuted_fold{args.fold}')

    # Save permuted model
    torch.save({
        'encoder_human': encoder_human_perm.state_dict(),
        'encoder_mouse': encoder_mouse_perm.state_dict(),
        'decoder_human': decoder_human_perm.state_dict(),
        'decoder_mouse': decoder_mouse_perm.state_dict(),
        'flow_h2m':      flow_h2m_perm.state_dict(),
        'perm_idx_human': perm_idx_human,
        'perm_idx_mouse': perm_idx_mouse,
        'args':vars(args),
    }, MODEL_DIR / f"fold_{args.fold}_permuted.pt")

    # Evaluate flow model on permuted validation data
    flow_h2m_perm.eval()
    with torch.no_grad():
        z_h2m_val_perm = validate_RNAseq_flowMatch_fold(device,
                                                        X_human_val, X_mouse_val,
                                                        encoder_human_perm, encoder_mouse_perm,
                                                        flow_h2m_perm,
                                                        translation_direction='1 to 2')

    # Compute permuted val latent reps separately for saving
    Z_human_val_perm = encode_all(encoder_human_perm, X_human_val, device)
    Z_mouse_val_perm = encode_all(encoder_mouse_perm, X_mouse_val, device)
    # Save validation latent variables
    np.save(MODEL_DIR / f"fold_{args.fold}_z_h2m_val_perm.npy",   z_h2m_val_perm.cpu().numpy())
    np.save(MODEL_DIR / f"fold_{args.fold}_z_human_val_perm.npy", Z_human_val_perm.numpy())
    np.save(MODEL_DIR / f"fold_{args.fold}_z_mouse_val_perm.npy", Z_mouse_val_perm.numpy())

    print2log(f"\n✓ Fold {args.fold} complete!")

if __name__ == "__main__":
    main()
