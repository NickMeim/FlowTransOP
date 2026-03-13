import os
import sys
import json
import signal
import argparse
import torch
from models import Decoder, SimpleEncoder, Flow
from trainingUtils import train_flowMatch_withSTRUCTURE_fold, validate_flowMatch_fold,train_AE_fold
from utility import *
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from logging import FileHandler

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = FileHandler('logs/AutoTransOP_Pretrain_FlowMatch_withSTRUCTURE.log', mode='a')
fh.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(fh)
print2log = logger.info

# Initialize argparse
parser = argparse.ArgumentParser(description="Run cross-fold validation for comparing with AutoTransOP.")
parser.add_argument('--checkpoint_dir', type=str, default='./chkpts_autotransop',
                    help='Directory to store checkpoints and progress.json')
parser.add_argument('--resume', type=int, default=1, help='1 to resume if checkpoint exists')
# Data and output paths
parser.add_argument('--folders', metavar='N', type=str, nargs='*', help='folders with paired datasets',
                    default=['A375_HT29', 'A375_PC3', 'HA1E_VCAP', 'HT29_MCF7', 'HT29_PC3', 'MCF7_HA1E', 'MCF7_PC3', 'PC3_HA1E'])
parser.add_argument('--data_root', type=str, help='Root directory for preprocessed data.',default='../preprocessing/preprocessed_data/CellPairs/')
parser.add_argument('--cmap_file', type=str, help='Path to the CMAP CSV file.',default='../preprocessing/preprocessed_data/CellPairs/drug_landmarks.csv')
parser.add_argument('--output_dir', type=str, help='Directory to save output results.',default='../results/AutoTransOP_CellPairs/')
# parser.add_argument('--pretrained_classifier', type=str, required=True, help='Path to the pre-trained classifier.')

# Training parameters
parser.add_argument('--batch_size_1', type=int, default=120, help='Batch size for dataset 1.')
parser.add_argument('--batch_size_2', type=int, default=120, help='Batch size for dataset 2.')
parser.add_argument('--batch_size_paired', type=int, default=80, help='Batch size for paired data.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

# Model parameters
parser.add_argument('--encoder_1_hiddens', type=int, nargs='+', default=[640, 384], help='Hidden layer sizes for encoder 1.')
parser.add_argument('--encoder_2_hiddens', type=int, nargs='+', default=[640, 384], help='Hidden layer sizes for encoder 2.')
parser.add_argument('--latent_dim', type=int, default=292, help='Dimension of the latent space.')
parser.add_argument('--decoder_1_hiddens', type=int, nargs='+', default=[384, 640], help='Hidden layer sizes for decoder 1.')
parser.add_argument('--decoder_2_hiddens', type=int, nargs='+', default=[384, 640], help='Hidden layer sizes for decoder 2.')
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
parser.add_argument('--V_dropout', type=float, default=0.25, help='Dropout rate for the species covariate.')
parser.add_argument('--state_class_hidden', type=int, nargs='+', default=[256, 128, 64], help='Hidden layer sizes for the state classifier.')
parser.add_argument('--state_class_drop_in', type=float, default=0.5, help='Input dropout rate for the state classifier.')
parser.add_argument('--state_class_drop', type=float, default=0.25, help='Dropout rate for the state classifier.')
parser.add_argument('--no_states', type=int, default=2, help='Number of states for the state classifier.')
parser.add_argument('--adv_class_hidden', type=int, nargs='+', default=[512, 256,128, 64,32,16], help='Hidden layer sizes for the adversarial classifier.')
parser.add_argument('--adv_class_drop_in', type=float, default=0.5, help='Input dropout rate for the adversarial classifier.')
parser.add_argument('--adv_class_drop', type=float, default=0.1, help='Dropout rate for the adversarial classifier.')
parser.add_argument('--no_adv_class', type=int, default=2, help='Number of classes for the adversarial classifier.')
parser.add_argument('--encoding_lr', type=float, default=0.001, help='Learning rate for the encoder.')
parser.add_argument('--adv_lr', type=float, default=0.001, help='Learning rate for the adversarial classifier.')
parser.add_argument('--schedule_step_adv', type=int, default=200, help='Step size for the adversarial learning rate scheduler.')
parser.add_argument('--gamma_adv', type=float, default=0.5, help='Gamma for the adversarial learning rate scheduler.')
parser.add_argument('--schedule_step_enc', type=int, default=200, help='Step size for the encoder learning rate scheduler.')
parser.add_argument('--gamma_enc', type=float, default=0.8, help='Gamma for the encoder learning rate scheduler.')
parser.add_argument('--no_folds', type=int, default=5, help='Number of cross-validation folds.')
parser.add_argument('--v_reg', type=float, default=1e-04, help='Regularization parameter for the species covariate.')
parser.add_argument('--state_class_reg', type=float, default=1e-02, help='Regularization parameter for the state classifier.')
parser.add_argument('--enc_l2_reg', type=float, default=0.01, help='L2 regularization for the encoder.')
parser.add_argument('--dec_l2_reg', type=float, default=0.01, help='L2 regularization for the decoder.')
parser.add_argument('--adv_penalnty', type=float, default=100, help='Penalty for the adversarial classifier.')
parser.add_argument('--reg_adv', type=float, default=1., help='Regularization for the adversarial classifier.')
parser.add_argument('--reg_classifier', type=float, default=1000, help='Regularization for the state classifier.')
parser.add_argument('--adversary_steps', type=int, default=4, help='Number of adversary steps.')
parser.add_argument('--autoencoder_wd', type=float, default=0.0, help='Weight decay for the autoencoder.')
parser.add_argument('--adversary_wd', type=float, default=0.0, help='Weight decay for the adversary.')
## arguments for stretching and alinging
parser.add_argument('--flow_lambda', type=float, default=1., help='Flow matrix regularization parameter.')
parser.add_argument('--conditional_flow_lambda', type=float, default=1e-3, help='Flow matching regularization parameter.')

args = parser.parse_args()
folders = args.folders
output_dir = args.output_dir
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

def _ensure_cols(df: pd.DataFrame, **defaults):
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v
    return df

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

def _save_metrics_state_atomic(state: dict, path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, 'w') as f:
        json.dump(state, f)
    os.replace(tmp, path)

def _set_metric(state: dict, name: str, fold_id: int, value):
    state[name][str(fold_id)] = float(value)

def _as_ordered_lists(state: dict):
    fold_ids = set()
    for m in state.values():
        fold_ids |= set(int(k) for k in m.keys())
    order = sorted(fold_ids)
    out = {}
    for metric, mp in state.items():
        out[metric] = [mp[str(fid)] for fid in order if str(fid) in mp]
    return out, order

# progress pointer (resume after last COMPLETED fold)
def save_progress(path: str, folder_idx: int, fold_id: int):
    state = {'last_completed': {'folder_idx': folder_idx, 'fold_id': fold_id}}
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

# ====== signal handling (place once) ============================================
_stop_requested = False
def _sigusr1_handler(signum, frame):
    # We only save at fold END. If this arrives mid-fold, we may redo that fold.
    global _stop_requested
    _stop_requested = True
    print2log('[Signal] USR1 received: will exit after current fold boundary.')
signal.signal(signal.SIGUSR1, _sigusr1_handler)

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
    'V_dropout': args.V_dropout,
    'state_class_hidden': args.state_class_hidden,
    'state_class_drop_in': args.state_class_drop_in,
    'state_class_drop': args.state_class_drop,
    'no_states': args.no_states,
    'adv_class_hidden': args.adv_class_hidden,
    'adv_class_drop_in': args.adv_class_drop_in,
    'adv_class_drop': args.adv_class_drop,
    'no_adv_class': args.no_adv_class,
    'encoding_lr': args.encoding_lr,
    'adv_lr': args.adv_lr,
    'schedule_step_adv': args.schedule_step_adv,
    'gamma_adv': args.gamma_adv,
    'schedule_step_enc': args.schedule_step_enc,
    'gamma_enc': args.gamma_enc,
    'batch_size_1': args.batch_size_1,
    'batch_size_2': args.batch_size_2,
    'batch_size_paired': args.batch_size_paired,
    'epochs': args.epochs,
    'no_folds': args.no_folds,
    'v_reg': args.v_reg,
    'state_class_reg': args.state_class_reg,
    'enc_l2_reg': args.enc_l2_reg,
    'dec_l2_reg': args.dec_l2_reg,
    'adv_penalnty': args.adv_penalnty,
    'reg_adv': args.reg_adv,
    'reg_classifier': args.reg_classifier,
    'adversary_steps': args.adversary_steps,
    'autoencoder_wd': args.autoencoder_wd,
    'adversary_wd': args.adversary_wd,
    'flow_lambda': args.flow_lambda,
    'conditional_flow_lambda': args.conditional_flow_lambda
}
class_criterion = torch.nn.CrossEntropyLoss()

progress_path = os.path.join(args.checkpoint_dir, 'progress.json')
Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

resume_ptr = {'folder_idx': -1, 'fold_id': 0}
if args.resume:
    p = load_progress(progress_path)
    if p and 'last_completed' in p:
        resume_ptr = p['last_completed']
        print2log(f"[Resume] Last completed: folder_idx={resume_ptr['folder_idx']} fold_id={resume_ptr['fold_id']}")

for fi, folder in enumerate(folders):
    if fi < resume_ptr['folder_idx']:
        continue
    # Extract dataset names from the folder name
    dataset1, dataset2 = folder.split('_')
    folder_path = os.path.join(args.data_root, folder)
    largest_sample_len = find_largest_sample_len(folder_path)
    print2log(f'Processing folder: {folder} with sample_len: {largest_sample_len}')

    # ---- per-folder cumulative CSVs (load existing or empty) ----
    latent_csv_path = os.path.join(output_dir, f"{folder}_flow12_TransAct_GeneralizedTransOP_latent_space_latent_withSTRUCTURE_eval.csv")
    trans_csv_path  = os.path.join(output_dir, f"{folder}_flow12_TransAct_GeneralizedTransOP_translation_withSTRUCTURE_eval.csv")
    df_latent_all = _load_or_empty_csv(latent_csv_path)
    df_trans_all  = _load_or_empty_csv(trans_csv_path)

    # ---- per-folder persistent simple metrics (for your lists) ----
    metrics_path = os.path.join(output_dir, f"{folder}__metrics.json")
    metrics_state = _load_metrics_state(metrics_path)
    _lists, _order = _as_ordered_lists(metrics_state)
    trainCosine = _lists.get("trainCosine", [])
    valCosine = _lists.get("valCosine", [])
    cosine_shuffledX = _lists.get("cosine_shuffledX", [])

    # You can keep df_result_1 etc. as scratch, they aren’t persisted:
    df_result_1 = pd.DataFrame({})
    df_result_2 = pd.DataFrame({})
    df_result_1_translation = pd.DataFrame({})
    df_result_2_translation = pd.DataFrame({})

    # Choose fold count from your params (recommended)
    total_folds = int(model_params.get('no_folds', 5))
    folds = range(1, total_folds + 1)
    for fold_id in folds:
        if (fi == resume_ptr['folder_idx']) and (fold_id <= resume_ptr['fold_id']):
            continue

        print2log(f"=== START fold {fold_id} in folder {folder} ===")
        # Example of loading data
        trainInfo_paired = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'train_paired_{fold_id}.csv'), index_col=None)
        trainInfo_1 = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'train_{dataset1}_{fold_id}.csv'), index_col=None)
        trainInfo_2 = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'train_{dataset2}_{fold_id}.csv'), index_col=None)

        valInfo_paired = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'val_paired_{fold_id}.csv'), index_col=None)
        valInfo_1 = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'val_{dataset1}_{fold_id}.csv'), index_col=None)
        valInfo_2 = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'val_{dataset2}_{fold_id}.csv'), index_col=None)

        # from the above metadata get x_train, y_train, x_val, y_val
        X_1 = torch.tensor(np.concatenate((cmap.loc[trainInfo_paired['sig_id.x']].values,
                                                 cmap.loc[trainInfo_1.sig_id].values))).float().to(device)
        X_2 = torch.tensor(np.concatenate((cmap.loc[trainInfo_paired['sig_id.y']].values,
                                           cmap.loc[trainInfo_2.sig_id].values))).float().to(device)
        # x for validation
        X_1_val = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,
                                                 cmap.loc[valInfo_1.sig_id].values))).float().to(device)
        X_2_val = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,
                                                 cmap.loc[valInfo_2.sig_id].values))).float().to(device)
        ## get for validation and training the pairs indices to acces them in the tensors
        pairs_train = np.arange(len(trainInfo_paired))
        pairs_val = np.arange(len(valInfo_paired))

        # Initialize data loaders

        # Initialize models for the fold
        decoder_1 = Decoder(model_params['latent_dim'], model_params['decoder_1_hiddens'], gene_size,
                            dropRate=model_params['dropout_decoder'], bn=model_params['bn_decoder'],
                            activation=model_params['decoder_activation'],dropIn=model_params['dropout_input_decoder']).to(device)
        decoder_2 = Decoder(model_params['latent_dim'], model_params['decoder_2_hiddens'], gene_size,
                            dropRate=model_params['dropout_decoder'], bn=model_params['bn_decoder'],
                            activation=model_params['decoder_activation'],dropIn=model_params['dropout_input_decoder']).to(device)
        encoder_1 = SimpleEncoder(gene_size, model_params['encoder_1_hiddens'], model_params['latent_dim'],
                                dropRate=model_params['dropout_encoder'], bn=model_params['bn_encoder'],
                                activation=model_params['encoder_activation'],dropIn=model_params['dropout_input_encoder']).to(device)
        encoder_2 = SimpleEncoder(gene_size, model_params['encoder_2_hiddens'], model_params['latent_dim'],
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
                                                      model_params['batch_size_1'], 
                                                      model_params['epochs'])
        (r2,decoder_2, encoder_2) = train_AE_fold(model_params, 
                                                      device, 
                                                      X_2,
                                                      decoder_2, 
                                                      encoder_2,
                                                      model_params['batch_size_2'], 
                                                      model_params['epochs'])
        print2log('Autoencoders training performance:')
        print2log(f'Fold {fold_id}: {r1}, {r2}')

        encoder_1.eval()
        encoder_2.eval()
        decoder_1.eval()
        decoder_2.eval()
        Z_1 = encoder_1(X_1.double())
        Z_2 = encoder_2(X_2.double())
        # Training and validation code here
        (pearson_1_to_2,cosine_train,flow_12) = train_flowMatch_withSTRUCTURE_fold(model_params, device,
                                                                     X_1,X_2,
                                                                     Z_1, Z_2,
                                                                     decoder_1, decoder_2,
                                                                     flow_12,
                                                                     model_params['batch_size_1'], model_params['batch_size_2'], model_params['epochs'],
                                                                     pairs_train,
                                                                     tanslation_direction = '1 to 2')
        
        # trainF1.append(f1)
        # trainAcc.append(class_acc)
        trainCosine.append(cosine_train)

        # Validation for the current fold
        r_1_to_2,pearson1,pearson2,cosine = validate_flowMatch_fold(device, X_1_val, X_2_val, decoder_1, decoder_2, encoder_1, encoder_2,flow_12,pairs_val,'1 to 2')
        mu_r = 0.5*(np.nanmean(pearson1) + np.nanmean(pearson2))
        # mu_r_translation = 0.5*(np.nanmean(r_1_to_2) + np.nanmean(r_2_to_1))
        mu_r_translation = np.nanmean(r_1_to_2)

        # print2log(f'Fold {fold_id}: F1 Score = {f1:.4f}, Class Accuracy = {class_acc:.4f}, r = {mu_r:.4f}, r_translation = {mu_r_translation:.4f}')
        print2log(f'Fold {fold_id}: r = {mu_r:.4f}, r_translation = {mu_r_translation:.4f}')

        # Train shuffled model for the current fold
        flow_12 = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2)).to(device)
        flow_21 = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2)).to(device)
        # flow_12 = ConditionalFlow(model_params['latent_dim'],model_params['latent_dim'],256).to(device)
        (_,_,flow_12) = train_flowMatch_withSTRUCTURE_fold(model_params, device,
                                                                     X_1, X_2,
                                                                     Z_1[:,np.random.permutation(Z_1.shape[1])], Z_2[:,np.random.permutation(Z_2.shape[1])],
                                                                     decoder_1, decoder_2,
                                                                     flow_12,
                                                                     model_params['batch_size_1'], model_params['batch_size_2'], model_params['epochs'],
                                                                     pairs_train,
                                                                     tanslation_direction = '1 to 2')
        # Validate the model for the current fold
        r_1_to_2_shuffledX, _, _,cosine_shuffled = validate_flowMatch_fold(device, X_1_val, X_2_val, decoder_1, decoder_2, encoder_1, encoder_2,flow_12,pairs_val,'1 to 2')

        # persist lists per-fold
        _set_metric(metrics_state, "trainCosine",      fold_id, cosine_train)
        _set_metric(metrics_state, "valCosine",        fold_id, cosine)
        _set_metric(metrics_state, "cosine_shuffledX", fold_id, cosine_shuffled)
        _save_metrics_state_atomic(metrics_state, metrics_path)
        trainCosine.append(cosine_train)
        valCosine.append(cosine)
        cosine_shuffledX.append(cosine_shuffled)

        # -------------------- PER-FOLD DATAFRAMES (THIS FOLD ONLY) --------------------
        # 1) Translation: arrays per column in one row
        tmp_trans = pd.DataFrame(
            {
                'train':      [pearson_1_to_2],           # can be list/np.array; will serialize via pandas
                'test':       [r_1_to_2],                 # same
                'shuffled X': [r_1_to_2_shuffledX],       # same
            },
            index=[fold_id]
        ).reset_index(drop=True)
        tmp_trans['fold'] = fold_id
        tmp_trans['translation'] = dataset1 + ' to ' + dataset2
        tmp_trans = _ensure_cols(tmp_trans, folder=folder, fold_id=fold_id)

        # 2) Latent-space: three rows for this fold (train/test/shuffled X) with scalar cosine
        tmp_latent = pd.DataFrame({
            'Cosine': [cosine_train, cosine, cosine_shuffled],
            'set':    ['train', 'test', 'shuffled X'],
            'fold':   [fold_id, fold_id, fold_id],
        })
        tmp_latent = _ensure_cols(tmp_latent, folder=folder, fold_id=fold_id)

        # -------------------- APPEND, DEDUP, ATOMIC WRITE -----------------------------
        latent_keys = ['folder', 'fold_id', 'set']   # one row per set per fold
        trans_keys  = ['folder', 'fold_id']          # one row per fold
        df_latent_all = _append_and_write_safely(df_latent_all, tmp_latent, latent_csv_path, dedup_subset=latent_keys)
        df_trans_all  = _append_and_write_safely(df_trans_all,  tmp_trans,  trans_csv_path,  dedup_subset=trans_keys)

        print2log(f'[Saved] {folder} fold={fold_id} latent/test={cosine:.4f} '
                  f'latent/train={cosine_train:.4f} latent/shuf={cosine_shuffled:.4f}')
        # If pearson_1_to_2 etc. are vectors, show their mean
        print2log(f'[Saved] {folder} fold={fold_id} translation r (mean) '
                  f'train={np.nanmean(pearson_1_to_2):.4f} '
                  f'test={np.nanmean(r_1_to_2):.4f} '
                  f'shuf={np.nanmean(r_1_to_2_shuffledX):.4f}')

        # -------------------- SAVE MODELS & PROGRESS AT FOLD END ----------------------
        save_models_at_fold_end(args.checkpoint_dir, tag='final', fold_id=fold_id,
                                encoder_1=encoder_1, decoder_1=decoder_1,
                                encoder_2=encoder_2, decoder_2=decoder_2,
                                flow_12=flow_12,
                                extra={'folder': folder})
        save_progress(progress_path, folder_idx=fi, fold_id=fold_id)
        print2log(f"=== END fold {fold_id} in folder {folder} ===")

        # Exit right after finishing the fold if USR1 arrived; Slurm will requeue
        if _stop_requested:
            print2log('[Signal] Exit after finishing fold; will resume with next fold on requeue.')
            sys.exit(0)
        
    print2log('Completely finished pair:'+folder)
