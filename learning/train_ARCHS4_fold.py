#!/usr/bin/env python3
"""
Training script for a single CV fold.
Submit as array job: sbatch --array=0-9 train_job.sh
"""
import argparse
from pathlib import Path
import torch
import numpy as np
from archs4_workflow import ARCHS4DataLoader, load_splits
from models import Decoder, SimpleEncoder, Flow, GaussianDecoder,VarDecoder,ElementWiseLinear
from trainingUtils import NBLoss,_convert_mean_disp_to_counts_logits, train_flowMatch_fold, validate_flowMatch_fold,train_count_AE_fold
from utility import *
from transact_utility_gpu import *
from evaluationUtils import pearson_r, r_square
import pandas as pd
import logging
from logging import FileHandler
import warnings
warnings.filterwarnings('ignore', message='.*ks_2samp.*')


# Configuration
DATA_DIR = Path("../archs4")  # Shared storage for downloaded files
SPLITS_DIR = Path("../archs4/splits")
MODEL_DIR = Path("../archs4/models")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for traiming.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--enc_l2_reg', type=float, default=1e-06, help='L2 regularization for the encoder.')
    parser.add_argument('--dec_l2_reg', type=float, default=1e-06, help='L2 regularization for the decoder.')
    parser.add_argument('--encoding_lr', type=float, default=0.001, help='Learning rate for the encoder.')
    parser.add_argument('--schedule_step_enc', type=int, default=8, help='Step size for the encoder learning rate scheduler.')
    parser.add_argument('--gamma_enc', type=float, default=0.8, help='Gamma for the encoder learning rate scheduler.')
    parser.add_argument('--autoencoder_wd', type=float, default=0.0, help='Weight decay for the autoencoder.')
    # Model parameters
    parser.add_argument('--encoder_1_hiddens', type=int, nargs='+', default=[4096,2048,1024,512], help='Hidden layer sizes for encoder 1.')
    parser.add_argument('--encoder_2_hiddens', type=int, nargs='+', default=[4096,2048,1024,512], help='Hidden layer sizes for encoder 2.')
    parser.add_argument('--latent_dim', type=int, default=512, help='Dimension of the latent space.')
    parser.add_argument('--decoder_1_hiddens', type=int, nargs='+', default=[512,768,2048, 4096], help='Hidden layer sizes for decoder 1.')
    parser.add_argument('--decoder_2_hiddens', type=int, nargs='+', default=[512,768,2048, 4096], help='Hidden layer sizes for decoder 2.')
    parser.add_argument('--dropout_decoder', type=float, default=0.1, help='Dropout rate for the decoder.')
    parser.add_argument('--dropout_encoder', type=float, default=0.1, help='Dropout rate for the encoder.')
    parser.add_argument('--bn_decoder', type=float, default=0.6, help='Use batch normalization in the decoder.')
    parser.add_argument('--bn_encoder', type=float, default=0.6, help='Use batch normalization in the encoder.')
    parser.add_argument('--dropout_input_encoder', type=float, default=0.5, help='Dropout rate for the imput of the encoder.')
    parser.add_argument('--dropout_input_decoder', type=float, default=0, help='Dropout rate for the imput of the decoder.')
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
        'no_folds': args.no_folds,
        'enc_l2_reg': args.enc_l2_reg,
        'dec_l2_reg': args.dec_l2_reg,
        'autoencoder_wd': args.autoencoder_wd,
        'flow_lambda': args.flow_lambda,
        'conditional_flow_lambda': args.conditional_flow_lambda
    }
    class_criterion = torch.nn.CrossEntropyLoss()
    
    # Load splits
    _, _, human_folds, mouse_folds = load_splits(SPLITS_DIR)
    
    # Initialize data loader
    loader = ARCHS4DataLoader(
        human_file=str(DATA_DIR / "human_gene_v2.latest.h5"),
        mouse_file=str(DATA_DIR / "mouse_gene_v2.latest.h5"),
        normalize=True,
        filter_genes=True
    )
    
    # Load fold data - returns DataFrames
    print(f"Loading fold {args.fold}...")
    human_train_df, human_val_df = loader.load_fold_data(
        human_folds[args.fold], "human"
    )
    mouse_train_df, mouse_val_df = loader.load_fold_data(
        mouse_folds[args.fold], "mouse"
    )
    
    # Convert to tensors - YOUR PATTERN # put batches to the gpu not all the data at once
    X_human = torch.tensor(human_train_df.values).float()#.to(device)
    X_mouse = torch.tensor(mouse_train_df.values).float()#.to(device)
    X_human_val = torch.tensor(human_val_df.values).float().to(device)
    X_mouse_val = torch.tensor(mouse_val_df.values).float().to(device)
    
    print(f"Human train: {X_human.shape}, val: {X_human_val.shape}")
    print(f"Mouse train: {X_mouse.shape}, val: {X_mouse_val.shape}")
    
    # ====== NORMAL MODEL ======
    print("\n=== Training NORMAL model ===")
    
    # Initialize models (your pattern)
    encoder_human = torch.nn.Sequential(ElementWiseLinear(X_human.shape[1]),
                                        SimpleEncoder(X_human.shape[1],
                                        model_params['encoder_1_hiddens'],
                                        model_params['latent_dim'],
                                        dropRate=model_params['dropout_encoder'],
                                        bn=model_params['bn_encoder'],
                                        activation=model_params['encoder_activation'],
                                        dropIn=model_params['dropout_input_encoder'])).to(device)
    encoder_mouse = torch.nn.Sequential(ElementWiseLinear(X_mouse.shape[1]),
                                        SimpleEncoder(X_mouse.shape[1],
                                        model_params['encoder_2_hiddens'],
                                        model_params['latent_dim'],
                                        dropRate=model_params['dropout_encoder'],
                                        bn=model_params['bn_encoder'],
                                        activation=model_params['encoder_activation'],
                                        dropIn=model_params['dropout_input_encoder'])).to(device)
    decoder_human = VarDecoder(model_params['latent_dim'],model_params['decoder_1_hiddens'],X_human.shape[1],
                           dropRate=model_params['dropout_decoder'],
                           activation=model_params['decoder_activation']).to(device)
    decoder_mouse = VarDecoder(model_params['latent_dim'],model_params['decoder_2_hiddens'],X_mouse.shape[1],
                           dropRate=model_params['dropout_decoder'],
                           activation=model_params['decoder_activation']).to(device)
    flow_h2m = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2),dtype=torch.float).to(device)
    # flow_m2h = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2),dtype=torch.float).to(device)
    
    # Train autoencoders - YOUR FUNCTIONS
    _, decoder_human, encoder_human = train_count_AE_fold(
        model_params, device, X_human, 
        decoder_human, encoder_human,
        model_params['batch_size'], model_params['epochs'],
        evaluate=False
    )
    _, decoder_mouse, encoder_mouse = train_count_AE_fold(
        model_params, device, X_mouse,
        decoder_mouse, encoder_mouse,
        model_params['batch_size'], model_params['epochs'],
        evaluate=False
    )

    encoder_human.eval()
    encoder_mouse.eval()
    decoder_human.eval()
    decoder_mouse.eval()
    with torch.no_grad():
        # Generate latent variables
        z_latent_base_1 = encoder_human(X_human_val)
        z_latent_base_2 = encoder_mouse(X_mouse_val)
        # reconstruction results
        y_pred_1 = decoder_human(z_latent_base_1)
        y_pred_2 = decoder_mouse(z_latent_base_2)
        # evaluate pearson correlation (pearson_r) of reconstruction
        r_human = pearson_r(y_pred_1, X_human_val).detach().cpu().numpy()
        r_mouse = pearson_r(y_pred_2, X_mouse_val).detach().cpu().numpy()

    print2log(f"Validation Reconstruction Pearson Correlation - Human: {np.nanmean(r_human):.4f}, Mouse: {np.nanmean(r_mouse):.4f}")
    
    # Get latent representations
    with torch.no_grad():
        Z_human = encoder_human(X_human.to(device))
        Z_mouse = encoder_mouse(X_mouse.to(device))
    
    # Train flow models
    _, _, flow_h2m = train_flowMatch_fold(
        model_params, device,
        X_human, X_mouse,
        Z_human, Z_mouse,
        decoder_human, decoder_mouse,
        flow_h2m,
        model_params['batch_size'], model_params['batch_size'], model_params['epochs'],
        tanslation_direction='1 to 2'
    )
    
    # Save normal model
    torch.save({
        'encoder_human': encoder_human.state_dict(),
        'encoder_mouse': encoder_mouse.state_dict(),
        'decoder_human': decoder_human.state_dict(),
        'decoder_mouse': decoder_mouse.state_dict(),
        'flow_h2m': flow_h2m.state_dict(),
    }, MODEL_DIR / f"fold_{args.fold}_normal.pt")
    
    
    # ====== PERMUTED MODEL ======
    print("\n=== Training PERMUTED model ===")
    
    # Reinitialize models
    encoder_human_perm = torch.nn.Sequential(ElementWiseLinear(X_human.shape[1]),
                                        SimpleEncoder(X_human.shape[1],
                                        model_params['encoder_1_hiddens'],
                                        model_params['latent_dim'],
                                        dropRate=model_params['dropout_encoder'],
                                        bn=model_params['bn_encoder'],
                                        activation=model_params['encoder_activation'],
                                        dropIn=model_params['dropout_input_encoder'])).to(device)
    encoder_mouse_perm = torch.nn.Sequential(ElementWiseLinear(X_mouse.shape[1]),
                                        SimpleEncoder(X_mouse.shape[1],
                                        model_params['encoder_2_hiddens'],
                                        model_params['latent_dim'],
                                        dropRate=model_params['dropout_encoder'],
                                        bn=model_params['bn_encoder'],
                                        activation=model_params['encoder_activation'],
                                        dropIn=model_params['dropout_input_encoder'])).to(device)
    decoder_human_perm = VarDecoder(model_params['latent_dim'],model_params['decoder_1_hiddens'],X_human.shape[1],
                           dropRate=model_params['dropout_decoder'],
                           activation=model_params['decoder_activation']).to(device)
    decoder_mouse_perm = VarDecoder(model_params['latent_dim'],model_params['decoder_2_hiddens'],X_mouse.shape[1],
                           dropRate=model_params['dropout_decoder'],
                           activation=model_params['decoder_activation']).to(device)
    
    flow_h2m_perm = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2), dtype=torch.float).to(device)
    # flow_m2h_perm = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2),dtype=torch.float).to(device)
    
    # Permute features
    perm_idx_human = torch.randperm(X_human.shape[1])
    perm_idx_mouse = torch.randperm(X_mouse.shape[1])
    X_human_permuted = X_human[:, perm_idx_human]
    X_mouse_permuted = X_mouse[:, perm_idx_mouse]
    
    # Train with permuted data
    _, decoder_human_perm, encoder_human_perm = train_count_AE_fold(
        model_params, device, X_human_permuted,
        decoder_human_perm, encoder_human_perm,
        model_params['batch_size'], model_params['epochs'],
        evaluate=False
    )
    _, decoder_mouse_perm, encoder_mouse_perm = train_count_AE_fold(
        model_params, device, X_mouse_permuted,
        decoder_mouse_perm, encoder_mouse_perm,
        model_params['batch_size'], model_params['epochs'],
        evaluate=False
    )

    encoder_human_perm.eval()
    encoder_mouse_perm.eval()
    decoder_human_perm.eval()
    decoder_mouse_perm.eval()
    with torch.no_grad():
        # Generate latent variables
        z_latent_base_1 = encoder_human_perm(X_human_val)
        z_latent_base_2 = encoder_mouse_perm(X_mouse_val)
        # reconstruction results
        y_pred_1 = decoder_human_perm(z_latent_base_1)
        y_pred_2 = decoder_mouse_perm(z_latent_base_2)
        # evaluate pearson correlation (pearson_r) of reconstruction
        r_human_perm = pearson_r(y_pred_1, X_human_val).detach().cpu().numpy()
        r_mouse_perm = pearson_r(y_pred_2, X_mouse_val).detach().cpu().numpy()

    print2log(f"Shuffled X Validation Reconstruction Pearson Correlation - Human: {np.nanmean(r_human_perm):.4f}, Mouse: {np.nanmean(r_mouse_perm):.4f}")
    
    # Get latent representations
    with torch.no_grad():
        Z_human_perm = encoder_human_perm(X_human_permuted)
        Z_mouse_perm = encoder_mouse_perm(X_mouse_permuted)
    
    pearson_h2m_perm, _, flow_h2m_perm = train_flowMatch_fold(
        model_params, device,
        X_human_permuted, X_mouse_permuted,
        Z_human_perm, Z_mouse_perm,
        decoder_human_perm, decoder_mouse_perm,
        flow_h2m_perm,
        model_params['batch_size'], model_params['batch_size'], model_params['epochs'],
        tanslation_direction='1 to 2'
    )
    
    # Save permuted model
    torch.save({
        'encoder_human': encoder_human_perm.state_dict(),
        'encoder_mouse': encoder_mouse_perm.state_dict(),
        'decoder_human': decoder_human_perm.state_dict(),
        'decoder_mouse': decoder_mouse_perm.state_dict(),
        'flow_h2m': flow_h2m_perm.state_dict(),
    }, MODEL_DIR / f"fold_{args.fold}_permuted.pt")
    
    print(f"Permuted model - Pearson: {pearson_h2m_perm:.4f}")
    
    # Save results summary
    results = {
        'fold': args.fold,
        'human_pearson': r_human_perm.tolist(),
        'mouse_pearson': r_mouse_perm.tolist(),
        'permuted_human_pearson': r_human_perm.tolist(),
        'permuted_mouse_pearson': r_mouse_perm.tolist()
    }
    
    import json
    with open(MODEL_DIR / f"fold_{args.fold}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Fold {args.fold} complete!")

if __name__ == "__main__":
    main()
