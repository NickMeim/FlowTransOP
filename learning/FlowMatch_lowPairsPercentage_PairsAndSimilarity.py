import os
import json
import argparse
import torch
from models import Decoder, SimpleEncoder, Flow
from trainingUtils import train_GeneralFM_fold, validate_flowMatch_fold,train_AE_fold
from utility import *
from transact_utility_gpu import *
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from logging import FileHandler
import warnings
warnings.filterwarnings('ignore', message='.*ks_2samp.*')

# Initialize argparse
parser = argparse.ArgumentParser(description="Run cross-fold validation for low pairs percentage for paired flow mathcing.")
parser.add_argument('--checkpoint_dir', type=str, default='./chkpts_PairAndSimilarity_lowPairsPercentage',
                    help='Directory to store checkpoints and progress.json')
parser.add_argument('--log_file',type=str,default='logs/PairAndSimilarity_lowPairsPercentage.log',help='Location of the log file that I can see print outputs')
parser.add_argument('--resume', type=int, default=1, help='1 to resume if checkpoint exists')
# Data and output paths
parser.add_argument('--folders', metavar='N', type=str, nargs='*', help='folders containing the data at different levels of percentages of pairs.',
                    default=['sample_ratio_114','sample_ratio_228','sample_ratio_342',
                             'sample_ratio_456','sample_ratio_569','sample_ratio_683'])
parser.add_argument('--data_root', type=str, help='Root directory for preprocessed data.',default='../preprocessing/preprocessed_data/pairedPercs/')
parser.add_argument('--cmap_file', type=str, help='Path to the CMAP CSV file.',default='../preprocessing/preprocessed_data/CellPairs/drug_landmarks.csv')
parser.add_argument('--output_dir', type=str, help='Directory to save output results.',default='../results/FlowMatch_fewPairs_A375_HT29_PairAndSimilarity/')
# Training parameters
parser.add_argument('--batch_size_1', type=int, default=120, help='Batch size for dataset 1.')
parser.add_argument('--batch_size_2', type=int, default=120, help='Batch size for dataset 2.')
parser.add_argument('--batch_size_paired', type=int, default=80, help='Batch size for paired data.')
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
## arguments for stretching and alinging
parser.add_argument('--flow_lambda', type=float, default=1., help='Flow matrix regularization parameter.')
parser.add_argument('--conditional_flow_lambda', type=float, default=1e-3, help='Flow matching regularization parameter.')
parser.add_argument(
    '--similarity_aggregation',
    type=str,
    choices=['max', 'mean', 'sum'],
    default='max',
    help=(
        'How to combine exact paired-condition indicators with TRANSACT-derived '
        'pre-aligned similarity. Default: max, the manuscript-selected hybrid.'
    ),
)
args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = FileHandler(args.log_file, mode='a')
fh.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(fh)
print2log = logger.info
print2log(f"Hybrid pair/similarity aggregation: {args.similarity_aggregation}")

output_dir = args.output_dir
data_root = args.data_root
folders = args.folders

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

# progress pointer (resume after last COMPLETED fold)
def save_progress(path: str,fold_id: int,folder_idx:int):
    state = {'last_completed': {'fold_id': fold_id,'folder_id':folder_idx}}
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
                            encoder_1, decoder_1, encoder_2, decoder_2, flow_12,flow_21, extra=None):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f'fold{fold_id}__{tag}.pt')
    payload = {
        'encoder_1': encoder_1.state_dict() if encoder_1 is not None else None,
        'decoder_1': decoder_1.state_dict() if decoder_1 is not None else None,
        'encoder_2': encoder_2.state_dict() if encoder_2 is not None else None,
        'decoder_2': decoder_2.state_dict() if decoder_2 is not None else None,
        'flow_12':   flow_12.state_dict()   if flow_12   is not None else None,
        'flow_21':   flow_21.state_dict()   if flow_21   is not None else None,
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
    "batch_size_1": args.batch_size_1,
    "batch_size_2": args.batch_size_2,
    "batch_size_paired": args.batch_size_paired,
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
progress_path = os.path.join(args.checkpoint_dir, 'progress.json')
Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

resume_ptr = {'fold_id': -1, 'folder_id': -1}
if args.resume:
    p = load_progress(progress_path)
    if p and 'last_completed' in p:
        resume_ptr = p['last_completed']
        print2log(f"[Resume] Last completed: fold_id={resume_ptr['fold_id']} itration_id={resume_ptr['folder_id']}")
for fi, folder in enumerate(folders):
    if fi < resume_ptr['fold_id']:
        continue
    print2log(f'Percentage of pairs used from folder: {folder}')
    ## define new batch size cause with actual pairs it struggles with bs_paired==80. Do the same as AutoTransOP original publication
    if folder == 'sample_ratio_114':
        model_params['batch_size_paired'] = 2
    elif folder == 'sample_ratio_228':
        model_params['batch_size_paired'] = 9
    elif folder == 'sample_ratio_342':
        model_params['batch_size_paired'] = 15
    elif folder == 'sample_ratio_456':
        model_params['batch_size_paired'] = 30
    elif folder == 'sample_ratio_569':
        model_params['batch_size_paired'] = 50
    elif folder == 'sample_ratio_683':
        model_params['batch_size_paired'] = 70
    # ---- per-folder cumulative CSVs (load existing or empty) ----
    recon_csv_path = os.path.join(output_dir, f"{folder}_flow_TransAct_GeneralizedTransOP_reconstruction_higherLR_moreReg_differentInputs.csv")
    trans_csv_path  = os.path.join(output_dir, f"{folder}_flow_TransAct_GeneralizedTransOP_translation_higherLR_moreReg_differentInputs.csv")
    df_recon_all = _load_or_empty_csv(recon_csv_path)
    df_trans_all  = _load_or_empty_csv(trans_csv_path)

    for fold_id in range(model_params['no_folds']):
        if (fi == resume_ptr['fold_id']) and (fold_id <= resume_ptr['fold_id']):
            continue
            
        print2log(f"=== Start fold {fold_id} for folder {fi+1}/{len(folders)} ===")
        # Example of loading data
        trainInfo_paired = pd.read_csv(data_root+folder+'/train_paired_'+str(fold_id+1)+'.csv', index_col=None)
        trainInfo_1 = pd.read_csv(data_root+folder+'/train_A375_'+str(fold_id+1)+'.csv', index_col=None)
        trainInfo_2 = pd.read_csv(data_root+folder+'/train_HT29_'+str(fold_id+1)+'.csv', index_col=None)

        valInfo_paired = pd.read_csv(data_root+folder+'/val_paired_'+str(fold_id+1)+'.csv', index_col=None)
        valInfo_1 = pd.read_csv(data_root+folder+'/val_A375_'+str(fold_id+1)+'.csv', index_col=None)
        valInfo_2 = pd.read_csv(data_root+folder+'/val_HT29_'+str(fold_id+1)+'.csv', index_col=None)
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
        print2log(f'Autoencoders for {folder} training performance:')
        print2log(f'{folder}: Fold {fold_id}: {r1}, {r2}')

        encoder_1.eval()
        encoder_2.eval()
        decoder_1.eval()
        decoder_2.eval()
        Z_1 = encoder_1(X_1.double())
        Z_2 = encoder_2(X_2.double())
        ## put all embeddings in a DataFrame
        all_emb1 = pd.DataFrame(encoder_1(torch.tensor(cmap.values,dtype=torch.double).to(device)).detach().cpu().numpy(),
                                index= cmap.index)
        all_emb2 = pd.DataFrame(encoder_2(torch.tensor(cmap.values,dtype=torch.double).to(device)).detach().cpu().numpy(),
                                index= cmap.index)
        # Training and validation code here
        print2log(f'Train flow for Fold {fold_id}: {folder}...')
        (pearson_1_to_2,_,flow_12) = train_GeneralFM_fold(model_params, device,
                                                          X_1,X_2,
                                                          Z_1, Z_2,
                                                          cmap,
                                                          all_emb1, all_emb2,trainInfo_1, trainInfo_2,trainInfo_paired,
                                                          decoder_1, decoder_2,
                                                          flow_12,
                                                          model_params['batch_size_1'], model_params['batch_size_2'],model_params['batch_size_paired'], model_params['epochs'],
                                                          pairs_train=pairs_train,
                                                          tanslation_direction = '1 to 2',
                                                          similarity_agregation = args.similarity_aggregation)
        print2log(f'Now train opposite flow for Fold {fold_id}: {folder}...')
        (pearson_2_to_1,_,flow_21) = train_GeneralFM_fold(model_params, device,
                                                          X_1, X_2,
                                                          Z_1, Z_2,
                                                          cmap,
                                                          all_emb1, all_emb2,trainInfo_1, trainInfo_2,trainInfo_paired,
                                                          decoder_1, decoder_2,
                                                          flow_21,
                                                          model_params['batch_size_1'], model_params['batch_size_2'],model_params['batch_size_paired'], model_params['epochs'],
                                                          pairs_train=pairs_train,
                                                          tanslation_direction = '2 to 1',
                                                          similarity_agregation = args.similarity_aggregation)
            
        # Validation for the current fold
        r_1_to_2,pearson1,_,_ = validate_flowMatch_fold(device, X_1_val, X_2_val, decoder_1, decoder_2, encoder_1, encoder_2,flow_12,pairs_val,'1 to 2')
        r_2_to_1,_,pearson2,_ = validate_flowMatch_fold(device, X_1_val, X_2_val, decoder_1, decoder_2, encoder_1, encoder_2,flow_21,pairs_val,'2 to 1')
        mu_r = 0.5*(np.nanmean(pearson1) + np.nanmean(pearson2))
        mu_r_translation = 0.5*(np.nanmean(r_1_to_2) + np.nanmean(r_2_to_1))
        print2log(f'Folder {fi+1}/{len(folders)}: Fold {fold_id}: r1 = {np.nanmean(pearson1):.4f}, r2 = {np.nanmean(pearson2):.4f}')
        print2log(f'Folder {fi+1}/{len(folders)}: Fold {fold_id}: r1_2 = {np.nanmean(r_1_to_2):.4f}, r2_1 = {np.nanmean(r_2_to_1):.4f}')

        # -------------------- PER-FOLD DATAFRAMES (THIS FOLD ONLY) --------------------
        # evaluate reconstruction
        tmp_recon = pd.DataFrame({'train':0.5*(r1+r2),'test':0.5*(pearson1+pearson2)},index=[fold_id])
        tmp_recon['fold'] = fold_id
        tmp_recon['folder'] = folder

        # 2) Evaluate translation
        tmp_trans = pd.DataFrame({'train':0.5*(pearson_1_to_2+pearson_2_to_1),'test':0.5*(r_1_to_2+r_2_to_1)},index=[fold_id])
        tmp_trans['fold'] = fold_id
        tmp_trans['folder'] = folder

        # -------------------- APPEND, DEDUP, ATOMIC WRITE -----------------------------
        recon_keys = ['folder', 'fold']   # one row per set per fold
        trans_keys  = ['folder', 'fold']          # one row per fold
        df_recon_all = _append_and_write_safely(df_recon_all, tmp_recon, recon_csv_path, dedup_subset=recon_keys)
        df_trans_all  = _append_and_write_safely(df_trans_all,  tmp_trans,  trans_csv_path,  dedup_subset=trans_keys)

        print2log(f'[Saved] {folder} fold={fold_id} reconstruction r (mean) '
                  f'train={np.nanmean(0.5*(pearson1+pearson2)):.4f} '
                  f'test={np.nanmean(0.5*(r1+r2)):.4f} ')
        print2log(f'[Saved] {folder} fold={fold_id} translation r (mean) '
                  f'train={np.nanmean(pearson_1_to_2):.4f} '
                  f'test={np.nanmean(r_1_to_2):.4f} ')

        # -------------------- SAVE MODELS & PROGRESS AT FOLD END ----------------------
        save_models_at_fold_end(args.checkpoint_dir, tag='final', fold_id=fold_id,
                                encoder_1=encoder_1, decoder_1=decoder_1,
                                encoder_2=encoder_2, decoder_2=decoder_2,
                                flow_12=flow_12,
                                flow_21=flow_21,
                                extra={'folder': folder, 'fold': fold_id})
        save_progress(progress_path, fold_id=fold_id,folder_idx=fi)
        print2log(f"=== END fold {fold_id} in folder {fi} out of {len(folders)} ===")
