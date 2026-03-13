import os
import json
import re
import argparse
import torch
from models import Decoder, SimpleEncoder, Flow
from trainingUtils import train_flowMatch_fold, validate_flowMatch_fold,train_AE_fold
from utility import *
from transact_utility_gpu import *
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from logging import FileHandler
import warnings
warnings.filterwarnings('ignore', message='.*ks_2samp.*')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = FileHandler('logs/AutoTransOP_Pretrain_FlowMatch_differentInputs_bracketed.log', mode='a')
fh.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(fh)
print2log = logger.info

# Initialize argparse
parser = argparse.ArgumentParser(description="Run cross-fold validation for different input features")
parser.add_argument('--checkpoint_dir', type=str, default='./chkpts_training_diffenetInputs_bracketed',
                    help='Directory to store checkpoints and progress.json')
parser.add_argument('--resume', type=int, default=1, help='1 to resume if checkpoint exists')
# Data and output paths
parser.add_argument('--cell_lines', metavar='N', type=str, nargs='*', help='cell lines to artificially create different inputs',
                    default=["PC3","HT29","MCF7","A549","NPC","HEPG2","A375","YAPC","U2OS","MCF10A","HA1E","HCC515","ASC","VCAP","HUVEC","HELA"])
parser.add_argument('--data_root', type=str, help='Root directory for preprocessed data.',default='../preprocessing/preprocessed_data/SameCellimputationModel/')
parser.add_argument('--cmap_file', type=str, help='Path to the CMAP CSV file.',default='../preprocessing/preprocessed_data/CellPairs/cmap_all_genes_q1_tas03.csv')
parser.add_argument('--output_dir', type=str, help='Directory to save output results.',default='../results/AutoTransOP_CellPairs_diffenetInputs/')
parser.add_argument('--brackets', metavar='N', type=str, nargs='*', help='Paths for the easier and harder brackets of differnt gene expression input',
                    default=["../preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/best_plus_same_features/high_correlation_set_1_iter0.npy",
                             "../preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/best_plus_same_features/high_correlation_set_1_iter1.npy",
                             "../preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/best_plus_same_features/high_correlation_set_1_iter2.npy",
                             "../preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/best_plus_same_features/high_correlation_set_1_iter3.npy",
                             "../preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/best_plus_same_features/high_correlation_set_1_iter4.npy",
                             "../preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/best_genes_",
                             "../preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/second_best_genes_",
                             "../preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/third_best_genes_",
                             "../preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/fourth_best_genes_"])
# Training parameters
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for traiming.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
parser.add_argument('--enc_l2_reg', type=float, default=0.01, help='L2 regularization for the encoder.')
parser.add_argument('--dec_l2_reg', type=float, default=0.01, help='L2 regularization for the decoder.')
parser.add_argument('--encoding_lr', type=float, default=0.001, help='Learning rate for the encoder.')
parser.add_argument('--schedule_step_enc', type=int, default=200, help='Step size for the encoder learning rate scheduler.')
parser.add_argument('--gamma_enc', type=float, default=0.8, help='Gamma for the encoder learning rate scheduler.')
parser.add_argument('--no_folds', type=int, default=5, help='Number of cross-validation folds.')
parser.add_argument('--autoencoder_wd', type=float, default=0.0, help='Weight decay for the autoencoder.')
# Model parameters
parser.add_argument('--encoder_1_hiddens', type=int, nargs='+', default=[384, 256], help='Hidden layer sizes for encoder 1.')
parser.add_argument('--encoder_2_hiddens', type=int, nargs='+', default=[384, 256], help='Hidden layer sizes for encoder 2.')
parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of the latent space.')
parser.add_argument('--decoder_1_hiddens', type=int, nargs='+', default=[256, 384], help='Hidden layer sizes for decoder 1.')
parser.add_argument('--decoder_2_hiddens', type=int, nargs='+', default=[256, 384], help='Hidden layer sizes for decoder 2.')
parser.add_argument('--dropout_decoder', type=float, default=0.2, help='Dropout rate for the decoder.')
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
cell_lines = args.cell_lines
brackets = args.brackets
output_dir = args.output_dir
data_root = args.data_root

Path(output_dir).mkdir(parents=True, exist_ok=True)
CKPT_DIR = args.checkpoint_dir
os.makedirs(CKPT_DIR, exist_ok=True)
PROGRESS = os.path.join(CKPT_DIR, 'progress.json')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print2log(f'Using device: {device}')

# Initialize environment and seeds for reproducibility
seed_everything(args.seed)

### Checkpoint helper functions
def _load_or_empty_csv(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def _atomic_to_csv(df: pd.DataFrame, path: str, **to_csv_kwargs):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False, **to_csv_kwargs)
    os.replace(tmp, path)

def _append_and_write_safely(existing_df: pd.DataFrame,
                             new_df: pd.DataFrame,
                             out_path: str,
                             dedup_subset=None):
    if new_df is None or (isinstance(new_df, pd.DataFrame) and new_df.empty):
        if not existing_df.empty and not os.path.exists(out_path):
            _atomic_to_csv(existing_df, out_path)
        return existing_df
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    if dedup_subset is not None and all(c in merged.columns for c in dedup_subset):
        merged = merged.drop_duplicates(subset=dedup_subset, keep='last')
    else:
        merged = merged.drop_duplicates(keep='last')
    _atomic_to_csv(merged, out_path)
    return merged

# Persist simple per-fold metrics so your lists survive requeues
def _load_metrics_state(path: str) -> dict:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except Exception:
                pass
    # structure: metric -> {fold_id(str): scalar}
    return {"trainCosine": {}, "valCosine": {}, "cosine_shuffledX": {}}

# progress pointer (resume after last COMPLETED fold)
def save_progress(path: str, cell_idx: int, fold_id: int):
    state = {'last_completed': {'cell_idx': cell_idx, 'fold_id': fold_id}}
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f)
    os.replace(tmp, path)

def load_progress(path: str):
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except Exception:
                return None
    return None

# Save models at fold end
def save_models_at_fold_end(ckpt_dir: str, tag: str, fold_id: int,
                            encoder_1, decoder_1, encoder_2, decoder_2, flow_12, extra=None):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f'fold{fold_id}__{tag}.pt')
    payload = {
        'encoder_1': encoder_1.state_dict() if encoder_1 is not None else None,
        'decoder_1': decoder_1.state_dict() if decoder_1 is not None else None,
        'encoder_2': encoder_2.state_dict() if encoder_2 is not None else None,
        'decoder_2': decoder_2.state_dict() if decoder_2 is not None else None,
        'flow_12':   flow_12.state_dict()   if flow_12   is not None else None,
        'extra': extra or {}
    }
    torch.save(payload, path)

# Read data
cmap = pd.read_csv(args.cmap_file, index_col=0)
genes = cmap.columns.values
gene_size = len(cmap.columns)
samples = cmap.index.values

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

df_result_translation = pd.DataFrame({})
df_result = pd.DataFrame({})
progress_path = os.path.join(args.checkpoint_dir, 'progress.json')
Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

resume_ptr = {'cell_idx': -1, 'fold_id': -1, 'iteration_id': -1}
if args.resume:
    p = load_progress(progress_path)
    if p and 'last_completed' in p:
        resume_ptr = p['last_completed']
        print2log(f"[Resume] Last completed: cell_idx={resume_ptr['cell_idx']} fold_id={resume_ptr['fold_id']} itration_id={resume_ptr['iteration_id']}")
for ci, cell in enumerate(cell_lines):
    if ci < resume_ptr['cell_idx']:
        continue
    print2log(f'Cell line to create artificially different inputs: {cell}')
    # ---- per-folder cumulative CSVs (load existing or empty) ----
    recon_csv_path = os.path.join(output_dir, f"{cell}_flow_TransAct_GeneralizedTransOP_reconstruction_higherLR_moreReg_differentInputs_bracketed.csv")
    trans_csv_path  = os.path.join(output_dir, f"{cell}_flow_TransAct_GeneralizedTransOP_translation_higherLR_moreReg_differentInputs_bracketed.csv")
    df_recon_all = _load_or_empty_csv(recon_csv_path)
    df_trans_all  = _load_or_empty_csv(trans_csv_path)

    # ---- per-folder persistent simple metrics (for your lists) ----
    metrics_path = os.path.join(output_dir, f"{cell}__metrics.json")
    metrics_state = _load_metrics_state(metrics_path)
    
    for j, bracket_path in enumerate(brackets):
        if (ci == resume_ptr['cell_idx']) and (j < resume_ptr['iteration_id']):
            continue
        if (re.search(r'high_correlation', bracket_path)):
            genes_1 = np.load(bracket_path,allow_pickle=True)
            genes_2 = np.load(bracket_path.replace('set_1','set_2'),allow_pickle=True)
        else:
            genes_1 = np.load(bracket_path + '1.npy',allow_pickle=True)
            genes_2 = np.load(bracket_path + '2.npy',allow_pickle=True)
        
        for fold_id in range(model_params['no_folds']):
            if (ci == resume_ptr['cell_idx']) and (j == resume_ptr['iteration_id']) and (fold_id <= resume_ptr['fold_id']):
                continue
            
            print2log(f"=== Start fold {fold_id} in iteration {j}/{len(brackets)} for cell {cell} ===")
            trainInfo = pd.read_csv(data_root+cell+'/train_'+str(fold_id)+'.csv',index_col=0)
            valInfo = pd.read_csv(data_root+cell+'/val_'+str(fold_id)+'.csv',index_col=0)
            if len(trainInfo)<950:
                bs = 256
            else:
                bs = model_params['batch_size']
            cmap_train = cmap.loc[trainInfo.sig_id,:]
            cols = cmap_train.columns.values
            cmap_val = cmap.loc[valInfo.sig_id,:]
            N = len(cmap_train)

            X_1 = torch.tensor(cmap_train.loc[:,genes_1].values).float().to(device)
            X_2 = torch.tensor(cmap_train.loc[:,genes_2].values).float().to(device)
            X_1_val = torch.tensor(cmap_val.loc[:,genes_1].values).float().to(device)
            X_2_val = torch.tensor(cmap_val.loc[:,genes_2].values).float().to(device)
            pairs_train = np.arange(X_1.shape[0])
            pairs_val = np.arange(X_1_val.shape[0])

            # Initialize models for the fold
            decoder_1 = Decoder(model_params['latent_dim'], model_params['decoder_1_hiddens'], X_1.shape[1],
                                dropRate=model_params['dropout_decoder'], bn=model_params['bn_decoder'],
                                activation=model_params['decoder_activation'],dropIn=model_params['dropout_input_decoder']).to(device)
            decoder_2 = Decoder(model_params['latent_dim'], model_params['decoder_2_hiddens'], X_2.shape[1],
                                dropRate=model_params['dropout_decoder'], bn=model_params['bn_decoder'],
                                activation=model_params['decoder_activation'],dropIn=model_params['dropout_input_decoder']).to(device)
            encoder_1 = SimpleEncoder(X_1.shape[1], model_params['encoder_1_hiddens'], model_params['latent_dim'],
                                    dropRate=model_params['dropout_encoder'], bn=model_params['bn_encoder'],
                                    activation=model_params['encoder_activation'],dropIn=model_params['dropout_input_encoder']).to(device)
            encoder_2 = SimpleEncoder(X_2.shape[1], model_params['encoder_2_hiddens'], model_params['latent_dim'],
                                    dropRate=model_params['dropout_encoder'],  bn=model_params['bn_encoder'],
                                    activation=model_params['encoder_activation'],dropIn=model_params['dropout_input_encoder']).to(device)
            flow_12 = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2)).to(device)
            flow_21 = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2)).to(device)
            # flow_12 = ConditionalFlow(model_params['latent_dim'],model_params['latent_dim'],256).to(device)

            ## First pretrain autoencoders for each biological context
            (r1,decoder_1, encoder_1) = train_AE_fold(model_params, 
                                                        device, 
                                                        X_1,
                                                        decoder_1, 
                                                        encoder_1,
                                                        bs, 
                                                        model_params['epochs'])
            (r2,decoder_2, encoder_2) = train_AE_fold(model_params, 
                                                        device, 
                                                        X_2,
                                                        decoder_2, 
                                                        encoder_2,
                                                        bs, 
                                                        model_params['epochs'])
            print2log(f'Autoencoders for {cell} training performance:')
            print2log(f'iteration {j}/{len(brackets)}: Fold {fold_id}: {r1}, {r2}')

            encoder_1.eval()
            encoder_2.eval()
            decoder_1.eval()
            decoder_2.eval()
            Z_1 = encoder_1(X_1.double())
            Z_2 = encoder_2(X_2.double())
            # Training and validation code here
            print2log(f'Train flow for iteration {j}/{len(brackets)}: Fold {fold_id}: {cell}...')
            (pearson_1_to_2,_,flow_12) = train_flowMatch_fold(model_params, device,
                                                                        X_1,X_2,
                                                                        Z_1, Z_2,
                                                                        decoder_1, decoder_2,
                                                                        flow_12,
                                                                        bs, bs, model_params['epochs'],
                                                                        pairs_train,
                                                                        tanslation_direction = '1 to 2')
            print2log(f'Now train opposite flow for iteration {j}/{len(brackets)}: Fold {fold_id}: {cell}...')
            (pearson_2_to_1,_,flow_21) = train_flowMatch_fold(model_params, device,
                                                                        X_1, X_2,
                                                                        Z_1, Z_2,
                                                                        decoder_1, decoder_2,
                                                                        flow_21,
                                                                        bs, bs, model_params['epochs'],
                                                                        pairs_train,
                                                                        tanslation_direction = '2 to 1')
            
            # Validation for the current fold
            r_1_to_2,pearson1,_,cosine = validate_flowMatch_fold(device, X_1_val, X_2_val, decoder_1, decoder_2, encoder_1, encoder_2,flow_12,pairs_val,'1 to 2')
            r_2_to_1,_,pearson2,cosine = validate_flowMatch_fold(device, X_1_val, X_2_val, decoder_1, decoder_2, encoder_1, encoder_2,flow_21,pairs_val,'2 to 1')
            mu_r = 0.5*(np.nanmean(pearson1) + np.nanmean(pearson2))
            mu_r_translation = 0.5*(np.nanmean(r_1_to_2) + np.nanmean(r_2_to_1))
            print2log(f'iteration {j}/{len(brackets)}: Fold {fold_id}: r1 = {np.nanmean(pearson1):.4f}, r2 = {np.nanmean(pearson2):.4f}')
            print2log(f'iteration {j}/{len(brackets)}: Fold {fold_id}: r1_2 = {np.nanmean(r_1_to_2):.4f}, r2_1 = {np.nanmean(r_2_to_1):.4f}')

            # Train shuffled model for the current fold
            flow_12 = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2)).to(device)
            flow_21 = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2)).to(device)
            print2log(f'Train shuffled X flow for iteration {j}/{len(brackets)}: Fold {fold_id}: {cell}...')
            (_,_,flow_12) = train_flowMatch_fold(model_params, device,
                                                                        X_1, X_2,
                                                                        Z_1[:,np.random.permutation(Z_1.shape[1])], Z_2[:,np.random.permutation(Z_2.shape[1])],
                                                                        decoder_1, decoder_2,
                                                                        flow_12,
                                                                        bs, bs, model_params['epochs'],
                                                                        pairs_train,
                                                                        tanslation_direction = '1 to 2')
            # print2log(f'Now train shuffled X opposite flow for iteration {j}/{len(brackets)}: Fold {fold_id}: {cell}...')
            # (_,_,flow_21) = train_flowMatch_fold(model_params, device,
            #                                                             X_2, X_1,
            #                                                             Z_2[:,np.random.permutation(Z_2.shape[1])], Z_1[:,np.random.permutation(Z_1.shape[1])],
            #                                                             decoder_2, decoder_1,
            #                                                             flow_21,
            #                                                             bs, bs, model_params['epochs'],
            #                                                             pairs_train,
            #                                                             tanslation_direction = '1 to 2')
            ## JUST SHOW SHUFFLED FOR ONE TRANSLATION DIRECTION ##
            # Validate the model for the current fold
            r_1_to_2_shuffledX, _, _,cosine = validate_flowMatch_fold(device, X_1_val, X_2_val, decoder_1, decoder_2, encoder_1, encoder_2,flow_12,pairs_val,'1 to 2')
            #r_2_to_1_shuffledX, _, _,cosine = validate_flowMatch_fold(device, X_1_val, X_2_val, decoder_1, decoder_2, encoder_1, encoder_2,flow_21,pairs_val,'2 to 1')

            # -------------------- PER-FOLD DATAFRAMES (THIS FOLD ONLY) --------------------
            # evaluate reconstruction
            tmp_recon = pd.DataFrame({'train':0.5*(r1+r2),'test':0.5*(pearson1+pearson2)},index=[fold_id])
            tmp_recon['fold'] = fold_id
            tmp_recon['cell'] = cell
            tmp_recon['iteration'] = j

            # 2) Latent-space: three rows for this fold (train/test/shuffled X) with scalar cosine
            tmp_trans = pd.DataFrame({'train':0.5*(pearson_1_to_2+pearson_2_to_1),'test':0.5*(r_1_to_2+r_2_to_1),'shuffled X':r_1_to_2_shuffledX},index=[fold_id])
            tmp_trans['fold'] = fold_id
            tmp_trans['cell'] = cell
            tmp_trans['iteration'] = j

            # -------------------- APPEND, DEDUP, ATOMIC WRITE -----------------------------
            recon_keys = ['cell', 'fold', 'iteration']   # one row per set per fold
            trans_keys  = ['cell', 'fold', 'iteration']          # one row per fold
            df_recon_all = _append_and_write_safely(df_recon_all, tmp_recon, recon_csv_path, dedup_subset=recon_keys)
            df_trans_all  = _append_and_write_safely(df_trans_all,  tmp_trans,  trans_csv_path,  dedup_subset=trans_keys)

            print2log(f'[Saved] {cell} fold={fold_id} iretation={j}/{len(brackets)} reconstruction r (mean) '
                      f'train={np.nanmean(0.5*(pearson1+pearson2)):.4f} '
                      f'test={np.nanmean(0.5*(r1+r2)):.4f} ')
            print2log(f'[Saved] {cell} fold={fold_id} iretation={j}/{len(brackets)} translation r (mean) '
                      f'train={np.nanmean(pearson_1_to_2):.4f} '
                      f'test={np.nanmean(r_1_to_2):.4f} '
                      f'shuffledX={np.nanmean(r_1_to_2_shuffledX):.4f}')

            # -------------------- SAVE MODELS & PROGRESS AT FOLD END ----------------------
            save_models_at_fold_end(args.checkpoint_dir, tag='final', fold_id=fold_id,
                                    encoder_1=encoder_1, decoder_1=decoder_1,
                                    encoder_2=encoder_2, decoder_2=decoder_2,
                                    flow_12=flow_12,
                                    extra={'cell': cell, 'fold': fold_id, 'iteration': j})
            save_progress(progress_path, cell_idx=ci, fold_id=fold_id)
            print2log(f"=== END fold {fold_id} in iteration {j}/{len(brackets)} for {cell} ===")
    print2log('=== Completely finished '+cell+' ===')
