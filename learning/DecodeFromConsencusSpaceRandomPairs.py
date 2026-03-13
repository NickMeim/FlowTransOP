import os
import argparse
import torch
from models import Decoder
from trainingUtils import train_decoder_fold
from evaluationUtils import pearson_r
from utility import *
from pathlib import Path
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

# Initialize argparse
parser = argparse.ArgumentParser(description="Run cross-fold validation for comparing with AutoTransOP.")
# Data and output paths
parser.add_argument('--folders', metavar='N', type=str, nargs='*', help='folders with paired datasets',
                    default=['A375_HT29', 'A375_PC3', 'HA1E_VCAP', 'HT29_MCF7', 'HT29_PC3', 'MCF7_HA1E', 'MCF7_PC3', 'PC3_HA1E'])
parser.add_argument('--data_root', type=str, help='Root directory for preprocessed data.',default='../preprocessing/preprocessed_data/CellPairs/')
parser.add_argument('--cmap_file', type=str, help='Path to the CMAP CSV file.',default='../preprocessing/preprocessed_data/CellPairs/drug_landmarks.csv')
parser.add_argument('--output_dir', type=str, help='Directory to save output results.',default='../results/AutoTransOP_CellPairs/')
# Training parameters
parser.add_argument('--batch_size_1', type=int, default=120, help='Batch size for dataset 1.')
parser.add_argument('--batch_size_2', type=int, default=120, help='Batch size for dataset 2.')
parser.add_argument('--batch_size_paired', type=int, default=80, help='Batch size for paired data.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
# Model parameters
parser.add_argument('--decoder_1_hiddens', type=int, nargs='+', default=[384, 640], help='Hidden layer sizes for decoder 1.')
parser.add_argument('--decoder_2_hiddens', type=int, nargs='+', default=[384, 640], help='Hidden layer sizes for decoder 2.')
parser.add_argument('--dropout_decoder', type=float, default=0.2, help='Dropout rate for the decoder.')
parser.add_argument('--bn_decoder', type=float, default=0.6, help='Use batch normalization in the decoder.')
parser.add_argument('--dropout_input_decoder', type=float, default=0, help='Dropout rate for the imput of the decoder.')
parser.add_argument('--decoder_activation', type=str,
                    choices=['LeakyReLU', 'ReLU', 'ELU', 'Sigmoid'],  
                    help='Activation function used between layers of the decoder',
                    default='ELU')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--schedule_step_enc', type=int, default=200, help='Step size for the encoder learning rate scheduler.')
parser.add_argument('--gamma_enc', type=float, default=0.8, help='Gamma for the encoder learning rate scheduler.')
parser.add_argument('--no_folds', type=int, default=10, help='Number of cross-validation folds.')
parser.add_argument('--dec_l2_reg', type=float, default=0.01, help='L2 regularization for the decoder.')
parser.add_argument('--autoencoder_wd', type=float, default=0.0, help='Weight decay for the autoencoder.')
parser.add_argument('--subset_size', type=int, default=128, help='Subset size for sampling some data to align randomly.')

args = parser.parse_args()
folders = args.folders
output_dir = args.output_dir
subset_size = args.subset_size

Path(output_dir).mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print2log(f'Using device: {device}')

# Initialize environment and seeds for reproducibility
seed_everything(args.seed)

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

# Model parameters
model_params = {
    'latent_dim': 0, # placeholder
    'decoder_1_hiddens': args.decoder_1_hiddens,
    'decoder_2_hiddens': args.decoder_2_hiddens,
    'dropout_decoder': args.dropout_decoder,
    'decoder_activation': decoder_activation,
    'bn_decoder': args.bn_decoder,
    'dropout_input_decoder': args.dropout_input_decoder,
    'lr': args.lr,
    'schedule_step_enc': args.schedule_step_enc,
    'gamma_enc': args.gamma_enc,
    'batch_size_1': args.batch_size_1,
    'batch_size_2': args.batch_size_2,
    'batch_size_paired': args.batch_size_paired,
    'epochs': args.epochs,
    'no_folds': args.no_folds,
    'dec_l2_reg': args.dec_l2_reg,
    'autoencoder_wd': args.autoencoder_wd
}
class_criterion = torch.nn.CrossEntropyLoss()

for folder in folders:
    # Extract dataset names from the folder name
    dataset1, dataset2 = folder.split('_')
    folder_path = os.path.join(args.data_root, folder)
    largest_sample_len = find_largest_sample_len(folder_path)
    print2log(f'Processing folder: {folder} with sample_len: {largest_sample_len}')

    # # Load pre-trained classifier
    # pretrained_adv_class = torch.load(args.pretrained_classifier)

    # Perform cross-fold validation
    trainF1 = []
    trainAcc = []
    valF1 = []
    valAcc = []
    valF1_shuffledX = []
    valAcc_shuffledX = []
    valCosine = []
    cosine_shuffledX = []
    trainCosine = []
    df_result_1_translation = pd.DataFrame({})
    df_result_2_translation = pd.DataFrame({})
    df_result_1 = pd.DataFrame({})
    df_result_2 = pd.DataFrame({})
    for fold_id in range(1, 6):
        # Example of loading data
        trainInfo_paired = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'train_paired_{fold_id}.csv'), index_col=None)
        trainInfo_1 = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'train_{dataset1}_{fold_id}.csv'), index_col=None)
        trainInfo_2 = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'train_{dataset2}_{fold_id}.csv'), index_col=None)

        valInfo_paired = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'val_paired_{fold_id}.csv'), index_col=None)
        valInfo_1 = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'val_{dataset1}_{fold_id}.csv'), index_col=None)
        valInfo_2 = pd.read_csv(os.path.join(folder_path, largest_sample_len, f'val_{dataset2}_{fold_id}.csv'), index_col=None)

        # from the above metadata get x_train, y_train, x_val, y_val
        X_1 = torch.tensor(np.concatenate((cmap.loc[trainInfo_paired['sig_id.x']].values,
                                                 cmap.loc[trainInfo_1.sig_id].values))).double().to(device)
        X_2 = torch.tensor(np.concatenate((cmap.loc[trainInfo_paired['sig_id.y']].values,
                                           cmap.loc[trainInfo_2.sig_id].values))).double().to(device)
        # x for validation
        X_1_val = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.x']].values,
                                                 cmap.loc[valInfo_1.sig_id].values))).double().to(device)
        X_2_val = torch.tensor(np.concatenate((cmap.loc[valInfo_paired['sig_id.y']].values,
                                                 cmap.loc[valInfo_2.sig_id].values))).double().to(device)
        ## get for validation and training the pairs indices to acces them in the tensors
        pairs_train = np.arange(len(trainInfo_paired))
        pairs_val = np.arange(len(valInfo_paired))

        for iter in range(10):
            x1_to_align = X_1[torch.randperm(X_1.shape[0])[:subset_size],:].detach().cpu().numpy() 
            x2_to_align = X_2[torch.randperm(X_2.shape[0])[:subset_size],:].detach().cpu().numpy()
            if (x1_to_align.shape[0]<subset_size):
                x_tmp = X_1[torch.randperm(X_1.shape[0])[:(subset_size-x1_to_align.shape[0])],:].detach().cpu().numpy()
                x1_to_align = np.concatenate((x1_to_align,x_tmp))
            if (x2_to_align.shape[0]<subset_size):
                x_tmp = X_2[torch.randperm(X_2.shape[0])[:(subset_size-x2_to_align.shape[0])],:].detach().cpu().numpy()
                x2_to_align = np.concatenate((x2_to_align,x_tmp))


            ## Align the samples in the space using only the training data, AS IF WE DID NOT HAVE PAIRED DATA
            _, _, _, intial_alignment_model = transact_align(
                x2_to_align,        # source → will become Z_source
                x1_to_align,        # target → will become Z_target
                n_src_pcs=75,
                n_tgt_pcs=75,
                n_pv=30,
                kernel='rbf',
                gamma=5e-4)
            X_2_aligned = transact_transform(X_2.detach().cpu().numpy(), intial_alignment_model,space='source')
            X_1_aligned = transact_transform(X_1.detach().cpu().numpy(), intial_alignment_model,space='target')
            
            model_params['latent_dim'] = X_2_aligned.shape[1]
            if (folder=='A375_HT29' and fold_id==1):
                print2log(f'latent_dim = {model_params["latent_dim"]}')
            ## Project validation data in the consencus space
            X_2_val_aligned = transact_transform(X_2_val.detach().cpu().numpy(), intial_alignment_model,space='source')
            X_1_val_aligned = transact_transform(X_1_val.detach().cpu().numpy(), intial_alignment_model,space='target')
            # Initialize models for the fold
            decoder_1 = Decoder(model_params['latent_dim'], model_params['decoder_1_hiddens'], gene_size,
                                dropRate=model_params['dropout_decoder'], bn=model_params['bn_decoder'],
                                activation=model_params['decoder_activation'],dropIn=model_params['dropout_input_decoder']).to(device)
            decoder_2 = Decoder(model_params['latent_dim'], model_params['decoder_2_hiddens'], gene_size,
                                dropRate=model_params['dropout_decoder'], bn=model_params['bn_decoder'],
                                activation=model_params['decoder_activation'],dropIn=model_params['dropout_input_decoder']).to(device)
            ## First pretrain autoencoders for each biological context
            (r1,decoder_1) = train_decoder_fold(model_params, 
                                                device, 
                                                X_1,
                                                torch.tensor(X_1_aligned, dtype=torch.double).to(device),
                                                decoder_1, 
                                                model_params['batch_size_1'], 
                                                model_params['epochs'])
            (r2,decoder_2) = train_decoder_fold(model_params, 
                                                device, 
                                                X_2,
                                                torch.tensor(X_2_aligned, dtype=torch.double).to(device),
                                                decoder_2, 
                                                model_params['batch_size_2'], 
                                                model_params['epochs'])
            # print2log('Autoencoders training performance:')
            # print2log(f'Fold {fold_id}: {r1}, {r2}')
            # Validation sets
            x_1_equivalent_val = torch.tensor(X_1_val_aligned[pairs_val,:], dtype=torch.double).to(device)
            x_2_equivalent_val = torch.tensor(X_2_val_aligned[pairs_val,:], dtype=torch.double).to(device)
            x_1_equivalent_train = torch.tensor(X_1_aligned[pairs_train,:], dtype=torch.double).to(device)
            x_2_equivalent_train = torch.tensor(X_2_aligned[pairs_train,:], dtype=torch.double).to(device)
            decoder_1.eval()
            decoder_2.eval()
            with torch.no_grad():
                ## training translation performance
                x_hat_2_equivalent_val = decoder_2(x_1_equivalent_train)
                x_hat_1_equivalent_val = decoder_1(x_2_equivalent_train)
                pearson_1_to_2 = pearson_r(x_hat_2_equivalent_val.flatten(), X_2[pairs_train,:].flatten()).detach().cpu().numpy()
                pearson_2_to_1 = pearson_r(x_hat_1_equivalent_val.flatten(), X_1[pairs_train,:].flatten()).detach().cpu().numpy()
                ## repeat for validation sets
                yhat2 = decoder_2(torch.tensor(X_2_val_aligned, dtype=torch.double).to(device))
                yhat1 = decoder_1(torch.tensor(X_1_val_aligned, dtype=torch.double).to(device))
                pearson1 = pearson_r(yhat1.flatten(), X_1_val.flatten()).detach().cpu().numpy()
                pearson2 = pearson_r(yhat2.flatten(), X_2_val.flatten()).detach().cpu().numpy()
                x_hat_2_equivalent_val = decoder_2(x_1_equivalent_val)
                x_hat_1_equivalent_val = decoder_1(x_2_equivalent_val)
                r_1_to_2 = pearson_r(x_hat_2_equivalent_val.flatten(), X_2_val[pairs_val,:].flatten()).detach().cpu().numpy()
                r_2_to_1 = pearson_r(x_hat_1_equivalent_val.flatten(), X_1_val[pairs_val,:].flatten()).detach().cpu().numpy()
                mu_r = 0.5*(np.nanmean(pearson1) + np.nanmean(pearson2))
                mu_r_translation = 0.5*(np.nanmean(r_1_to_2) + np.nanmean(r_2_to_1))
            print2log(f'Fold {fold_id} / Iteration {iter}: r = {mu_r:.4f}, r_translation = {mu_r_translation:.4f}')

            # evaluate reconstruction
            tmp = pd.DataFrame({'train':r1,'test':pearson1},index=[fold_id])
            tmp['subset_size'] = subset_size
            tmp['fold'] = fold_id
            tmp['iteration'] = iter
            tmp['context'] = dataset1
            df_result_1 = pd.concat([df_result_1, tmp], axis=0)
            tmp = pd.DataFrame({'train':r2,'test':pearson2},index=[fold_id])
            tmp['subset_size'] = subset_size
            tmp['fold'] = fold_id
            tmp['iteration'] = iter
            tmp['context'] = dataset2
            df_result_2 = pd.concat([df_result_2, tmp], axis=0)
            df_result = pd.concat([df_result_1, df_result_2], axis=0)
            df_result.to_csv(output_dir+folder+'_Decoder_TransActPaired_reconstruction_eval_subsetted_'+str(subset_size)+'.csv')
            # repeat just for translation
            tmp = pd.DataFrame({'train':pearson_1_to_2,'test':r_1_to_2},index=[fold_id])
            tmp['subset_size'] = subset_size
            tmp['fold'] = fold_id
            tmp['iteration'] = iter
            tmp['translation'] = dataset1+' to '+dataset2
            df_result_1_translation = pd.concat([df_result_1_translation, tmp], axis=0)
            tmp = pd.DataFrame({'train':pearson_2_to_1,'test':r_2_to_1},index=[fold_id])
            tmp['subset_size'] = subset_size
            tmp['fold'] = fold_id
            tmp['iteration'] = iter
            tmp['translation'] = dataset2+' to '+dataset1
            df_result_2_translation = pd.concat([df_result_2_translation, tmp], axis=0)
            df_result_translation = pd.concat([df_result_1_translation, df_result_2_translation], axis=0)
            df_result_translation.to_csv(output_dir+folder+'_Decoder_TransActPaired_translation_eval_'+str(subset_size)+'.csv')
        
    print2log('Completely finished pair:'+folder)
