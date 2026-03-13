import os
import argparse
import torch
from models import Decoder, SimpleEncoder, Flow, ConditionalFlow
from trainingUtils import train_flowMatch_withpairs_fold, validate_flowMatch_fold,train_AE_fold
from utility import *
from pathlib import Path
import numpy as np
import pandas as pd
import logging

def _dedupe_mean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.astype(str).str.strip()
    return out.groupby(level=0, sort=False).mean()

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
# parser.add_argument('--prior_beta', type=float, default=1.0, help='Beta parameter for the prior discriminator.')
parser.add_argument('--no_folds', type=int, default=10, help='Number of cross-validation folds.')
parser.add_argument('--v_reg', type=float, default=1e-04, help='Regularization parameter for the species covariate.')
parser.add_argument('--state_class_reg', type=float, default=1e-02, help='Regularization parameter for the state classifier.')
parser.add_argument('--enc_l2_reg', type=float, default=0.01, help='L2 regularization for the encoder.')
parser.add_argument('--dec_l2_reg', type=float, default=0.01, help='L2 regularization for the decoder.')
## arguments for stretching and alinging
parser.add_argument('--beta_stretch', type=float, default=100., help='Beta parameter for L_stretch.')
parser.add_argument('--beta_align', type=float, default=10., help='Beta parameter for L_align.')
parser.add_argument('--power_iteration_steps', type=int, default=100, help='Number of power iteration steps to estimate maximum eigenvalue.')
parser.add_argument('--flow_lambda', type=float, default=1., help='Flow matrix regularization parameter.')
parser.add_argument('--conditional_flow_lambda', type=float, default=1e-3, help='Flow matching regularization parameter.')

args = parser.parse_args()
folders = args.folders
output_dir = args.output_dir

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
    'beta_stretch': args.beta_stretch,
    'beta_align': args.beta_align,
    'power_iteration_steps': args.power_iteration_steps,
    'flow_lambda': args.flow_lambda,
    'conditional_flow_lambda': args.conditional_flow_lambda
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
    df_result_1 = pd.DataFrame({})
    df_result_2 = pd.DataFrame({})
    df_result_1_translation = pd.DataFrame({})
    df_result_2_translation = pd.DataFrame({})
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
        ## put all embeddings in a DataFrame
        all_emb1 = pd.DataFrame(encoder_1(torch.tensor(cmap.values,dtype=torch.double).to(device)).detach().cpu().numpy(),
                                index= cmap.index)
        all_emb2 = pd.DataFrame(encoder_2(torch.tensor(cmap.values,dtype=torch.double).to(device)).detach().cpu().numpy(),
                                index= cmap.index)

        # Training and validation code here
        (pearson_1_to_2,cosine_train,flow_12) = train_flowMatch_withpairs_fold(model_params, device,
                                                                     X_1,X_2,
                                                                     Z_1, Z_2,
                                                                     all_emb1, all_emb2,trainInfo_1, trainInfo_2,trainInfo_paired,
                                                                     encoder_1, encoder_2,
                                                                     decoder_1, decoder_2,
                                                                     flow_12,
                                                                     model_params['batch_size_1'], model_params['batch_size_2'], model_params['batch_size_paired'], model_params['epochs'],
                                                                     pairs_train,
                                                                     tanslation_direction = '1 to 2')
        
        # trainF1.append(f1)
        # trainAcc.append(class_acc)
        trainCosine.append(cosine_train)

        # Validation for the current fold
        r_1_to_2,pearson1,pearson2,cosine = validate_flowMatch_fold(device, X_1_val, X_2_val, decoder_1, decoder_2, encoder_1, encoder_2,flow_12,pairs_val,'1 to 2')
        # valF1.append(f1)
        # valAcc.append(class_acc)
        valCosine.append(cosine)
        mu_r = 0.5*(np.nanmean(pearson1) + np.nanmean(pearson2))
        # mu_r_translation = 0.5*(np.nanmean(r_1_to_2) + np.nanmean(r_2_to_1))
        mu_r_translation = np.nanmean(r_1_to_2)

        # print2log(f'Fold {fold_id}: F1 Score = {f1:.4f}, Class Accuracy = {class_acc:.4f}, r = {mu_r:.4f}, r_translation = {mu_r_translation:.4f}')
        print2log(f'Fold {fold_id}: r = {mu_r:.4f}, r_translation = {mu_r_translation:.4f}')

        # Train shuffled model for the current fold
        flow_12 = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2)).to(device)
        flow_21 = Flow(model_params['latent_dim'], int(model_params['latent_dim']/2)).to(device)
        # flow_12 = ConditionalFlow(model_params['latent_dim'],model_params['latent_dim'],256).to(device)
        (_,_,flow_12) = train_flowMatch_withpairs_fold(model_params, device,
                                                                     X_1, X_2,
                                                                     Z_1, Z_2,
                                                                     all_emb1.iloc[:,np.random.permutation(Z_1.shape[1])], all_emb2.iloc[:,np.random.permutation(Z_2.shape[1])],
                                                                     trainInfo_1, trainInfo_2,trainInfo_paired,
                                                                     encoder_1, encoder_2,
                                                                     decoder_1, decoder_2,
                                                                     flow_12,
                                                                     model_params['batch_size_1'], model_params['batch_size_2'], model_params['batch_size_paired'], model_params['epochs'],
                                                                     pairs_train,
                                                                     tanslation_direction = '1 to 2')
        # Validate the model for the current fold
        r_1_to_2_shuffledX, _, _,cosine = validate_flowMatch_fold(device, X_1_val, X_2_val, decoder_1, decoder_2, encoder_1, encoder_2,flow_12,pairs_val,'1 to 2')
        # valF1_shuffledX.append(f1)
        # valAcc_shuffledX.append(class_acc)
        cosine_shuffledX.append(cosine)

        # # put in reconstruction results in data frame
        # tmp = pd.DataFrame({'train':pear_train_1,'test':pearson1,'shuffled X':pearson1_shuffledX},index=[fold_id])
        # tmp['fold'] = fold_id
        # tmp['system'] = dataset1
        # # tmp['gene'] = genes
        # df_result_1 = pd.concat([df_result_1, tmp], axis=0)
        # tmp = pd.DataFrame({'train':pear_train_2,'test':pearson2,'shuffled X':pearson2_shuffledX},index=[fold_id])
        # tmp['fold'] = fold_id
        # tmp['system'] = dataset2
        # # tmp['gene'] = genes
        # df_result_2 = pd.concat([df_result_2, tmp], axis=0)
        # df_result = pd.concat([df_result_1, df_result_2], axis=0)
        # df_result.to_csv(output_dir+folder+'_flow_GeneralizedTransOP_reconstruction_eval.csv')

        # repeat just for translation
        tmp = pd.DataFrame({'train':pearson_1_to_2,'test':r_1_to_2,'shuffled X':r_1_to_2_shuffledX},index=[fold_id])
        tmp['fold'] = fold_id
        tmp['translation'] = dataset1+' to '+dataset2
        # tmp['gene'] = genes
        df_result_1_translation = pd.concat([df_result_1_translation, tmp], axis=0)
        # tmp = pd.DataFrame({'train':pearson_2_to_1,'test':r_2_to_1,'shuffled X':r_2_to_1_shuffledX},index=[fold_id])
        # tmp['fold'] = fold_id
        # tmp['translation'] = dataset2+' to '+dataset1
        # # tmp['gene'] = genes
        # df_result_2_translation = pd.concat([df_result_2_translation, tmp], axis=0)
        # df_result_translation = pd.concat([df_result_1_translation, df_result_2_translation], axis=0)
        # df_result_translation.to_csv(output_dir+folder+'_flow_GeneralizedTransOP_translation_eval.csv')
        df_result_1_translation.to_csv(output_dir+folder+'_flow12_TransAct_GeneralizedTransOP_withPairs_translation_eval.csv')

        # save also latent space performance
        # valdiation performance
        # df_result_latent_test = pd.DataFrame({'F1':valF1,'Accuracy':valAcc,'Cosine':valCosine})
        df_result_latent_test = pd.DataFrame({'Cosine':valCosine})
        df_result_latent_test['fold'] = list(range(fold_id))
        df_result_latent_test['set'] = 'test'
        #train performance
        # df_result_latent_train = pd.DataFrame({'F1':trainF1,'Accuracy':trainAcc})
        df_result_latent_train = pd.DataFrame({'Cosine':trainCosine})
        df_result_latent_train['fold'] = list(range(fold_id))
        df_result_latent_train['set'] = 'train'
        # shuffled X performance
        # df_result_latent_shuffledX = pd.DataFrame({'F1':valF1_shuffledX,'Accuracy':valAcc_shuffledX})
        df_result_latent_shuffledX = pd.DataFrame({'Cosine':cosine_shuffledX})
        df_result_latent_shuffledX['fold'] = list(range(fold_id))
        df_result_latent_shuffledX['set'] = 'shuffled X'
        # combine
        df_result_latent = pd.concat([df_result_latent_train,df_result_latent_test,df_result_latent_shuffledX], axis=0)
        df_result_latent.to_csv(output_dir+folder+'_flow12_TransAct_GeneralizedTransOP_withPairs_latent_eval.csv')
        
    print2log('Completely finished pair:'+folder)
