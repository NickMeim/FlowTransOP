import numpy as np
import torch
import pandas as pd
import math
from evaluationUtils import pearson_r, r_square
from utility import partitionAndMatch
from sklearn.metrics import confusion_matrix
from geomloss import SamplesLoss
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

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
    return outList

def compute_gradients(output, input):
    grads = torch.autograd.grad(output, input, create_graph=True)
    grads = grads[0].pow(2).mean()
    return grads

def contrastive_loss(x, y, margin=1.0):
    pairwise_dist = torch.mean(torch.cdist(x, y))
    return pairwise_dist

def validate_fold(device, x_1_test, x_2_test,
                              decoder_1, decoder_2, encoder_1, encoder_2,classifier, Vsp,pairs_val=None): #encoder_2, classifier, Vsp,pairs_val=None
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
        # evaluate pearson correlation (pearson_r) of reconstruction
        r_1 = pearson_r(y_pred_1, x_1).detach().cpu().numpy()
        r_2 = pearson_r(y_pred_2, x_2).detach().cpu().numpy()

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
            pearson_1_to_2 = pearson_r(x_hat_2_equivalent, x_2_equivalent).detach().cpu().numpy()
            # second translate system 2 to 1
            z_latent_base_2_equivalent  = encoder_2(x_2_equivalent)
            z_latent_2_equivalent = Vsp(z_latent_base_2_equivalent,1.-z_species_2_equivalent)
            x_hat_1_equivalent = decoder_1(z_latent_2_equivalent).detach()
            pearson_2_to_1 = pearson_r(x_hat_1_equivalent, x_1_equivalent).detach().cpu().numpy()
        else:
            pearson_1_to_2 = np.nan
            pearson_2_to_1 = np.nan
    
    return pearson_1_to_2, pearson_2_to_1,f1, class_acc, r_1, r_2 #pearson_2_to_1,f1, class_acc

def train_fold(model_params, device, x_1_train, x_2_train,
                    decoder_1, decoder_2, encoder_1, encoder_2, adverse_classifier,# classifier, Vsp,
                    bs_1:int, bs_2:int, NUM_EPOCHS:int,
                    class_criterion,
                    pairs_train=None):    
    # Combine parameters and create optimizers/schedulers
    allParams = list(decoder_1.parameters()) + list(decoder_2.parameters()) + \
                list(encoder_1.parameters()) + list(encoder_2.parameters()) #+ \
                #list(classifier.parameters()) + list(Vsp.parameters())
    
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
        #classifier.train()
        adverse_classifier.train()
        #Vsp.train()

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
            #z_species_1 = torch.cat((torch.ones(X_1.shape[0], 1), torch.zeros(X_1.shape[0], 1)), 1).double().to(device)
            #z_species_2 = torch.cat((torch.zeros(X_2.shape[0], 1), torch.ones(X_2.shape[0], 1)), 1).double().to(device)
            
            optimizer.zero_grad()
            optimizer_adv.zero_grad()
                        
            if e==0:#e % model_params['schedule_step_adv'] == 0:
                for _ in range(20):
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
                optimizer_adv.zero_grad()
                for _ in range(5):
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
                # now perform the non-aversesary step    
                optimizer.zero_grad()
                z_base_1 = encoder_1(X_1)
                z_base_2 = encoder_2(X_2)
                latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
                
                #z_1 = Vsp(z_base_1,z_species_1)
                #z_2 = Vsp(z_base_2,z_species_2)
                #latent_vectors = torch.cat((z_1, z_2), 0)
                
                #y_pred_1 = decoder_1(z_1)
                y_pred_1 = decoder_1(z_base_1)
                fitLoss_1 = torch.mean(torch.sum((y_pred_1 - X_1)**2,dim=1))
                L2Loss_1 = decoder_1.L2Regularization(model_params['dec_l2_reg']) + encoder_1.L2Regularization(model_params['enc_l2_reg'])
                loss_1 = fitLoss_1 + L2Loss_1
                
                #y_pred_2 = decoder_2(z_2)
                y_pred_2 = decoder_2(z_base_2)
                fitLoss_2 = torch.mean(torch.sum((y_pred_2 - X_2)**2,dim=1))
                L2Loss_2 = decoder_2.L2Regularization(model_params['dec_l2_reg']) + encoder_2.L2Regularization(model_params['enc_l2_reg'])
                loss_2 = fitLoss_2 + L2Loss_2

                # Classification loss
                #labels = classifier(latent_vectors)
                #true_labels = torch.cat((torch.ones(z_1.shape[0]),
                #    torch.zeros(z_2.shape[0])),0).long().to(device)
                #entropy = class_criterion(labels,true_labels)
                #_, predicted = torch.max(labels, 1)
                #predicted = predicted.cpu().numpy()
                #cf_matrix = confusion_matrix(true_labels.cpu().numpy(),predicted)
                #tn, fp, fn, tp = cf_matrix.ravel()
                #f1_latent = 2*tp/(2*tp+fp+fn)
                
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
                
                loss = loss_1 + loss_2+ model_params['distance_reg']*dist_loss - model_params['reg_adv']*adv_entropy #+model_params['reg_classifier'] * entropy+classifier.L2Regularization(model_params['state_class_reg']) +Vsp.Regularization(model_params['v_reg'])
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
            #outString += ', Entropy Loss={:.4f}'.format(entropy.item())
            outString += ', Adverse Entropy={:.4f}'.format(adv_entropy.item())
            outString += ', loss={:.4f}'.format(loss.item())
            #outString += ', F1 latent={:.4f}'.format(f1_latent)
            outString += ', F1 basal={:.4f}'.format(f1_basal)
            outString += ', F1 basal trained={:.4f}'.format(f1_basal_trained)
            outString += ',distance loss={:.4f}'.format(dist_loss.item())

        # Logging
        if (e % 250 == 0 and e > 0) or (e == 1) or (e + 1 == NUM_EPOCHS):
            print2log(outString)
        
    # evaluate
    encoder_1.eval()
    encoder_2.eval()
    decoder_1.eval()
    decoder_2.eval()
    #classifier.eval()
    #Vsp.eval()
    with torch.no_grad():
        z_base_1 = encoder_1(x_1_train.double().to(device))
        z_base_2 = encoder_2(x_2_train.double().to(device))
        #z_species_1 = torch.cat((torch.ones(x_1_train.shape[0], 1), torch.zeros(x_1_train.shape[0], 1)), 1).double().to(device)
        #z_species_2 = torch.cat((torch.zeros(x_2_train.shape[0], 1), torch.ones(x_2_train.shape[0], 1)), 1).double().to(device)
        #z_1 = Vsp(z_base_1,z_species_1)
        #z_2 = Vsp(z_base_2,z_species_2)
        #y_pred_1 = decoder_1(z_1)
        #y_pred_2 = decoder_2(z_2)
        y_pred_1 = decoder_1(z_base_1)
        y_pred_2 = decoder_2(z_base_2)
        pear_1 = pearson_r(y_pred_1.detach(), x_1_train.double().to(device)).cpu().numpy()
        pear_2 = pearson_r(y_pred_2.detach(), x_2_train.double().to(device)).cpu().numpy()
        # Classification results
        #labels = classifier(torch.cat((z_1, z_2), 0))
        #true_labels = torch.cat((torch.ones(z_1.shape[0]).view(z_1.shape[0], 1),
        #    torch.zeros(z_2.shape[0]).view(z_2.shape[0], 1)), 0).long()
        #_, predicted = torch.max(labels, 1)
        #predicted = predicted.cpu().numpy()
        #cf_matrix = confusion_matrix(true_labels.numpy(), predicted)
        #tn, fp, fn, tp = cf_matrix.ravel()
        #class_acc = (tp + tn) / predicted.size
        #f1 = 2 * tp / (2 * tp + fp + fn)

        # translation results
        if pairs_train is not None:
            x_1_equivalent = x_1_train[pairs_train,:].double()
            x_2_equivalent = x_2_train[pairs_train,:].double()
            #z_species_1_equivalent = z_species_1[pairs_train,:]
            #z_species_2_equivalent = z_species_2[pairs_train,:]
            # first translate system 1 to 2
            z_latent_base_1_equivalent  = encoder_1(x_1_equivalent)
            #z_latent_1_equivalent = Vsp(z_latent_base_1_equivalent,1.-z_species_1_equivalent)
            #x_hat_2_equivalent = decoder_2(z_latent_1_equivalent).detach()
            x_hat_2_equivalent = decoder_2(z_latent_base_1_equivalent).detach()
            pearson_1_to_2 = pearson_r(x_hat_2_equivalent, x_2_equivalent).detach().cpu().numpy()
            # second translate system 2 to 1
            z_latent_base_2_equivalent  = encoder_2(x_2_equivalent)
            #z_latent_2_equivalent = Vsp(z_latent_base_2_equivalent,1.-z_species_2_equivalent)
            #x_hat_1_equivalent = decoder_1(z_latent_2_equivalent).detach()
            x_hat_1_equivalent = decoder_1(z_latent_base_2_equivalent).detach()
            pearson_2_to_1 = pearson_r(x_hat_1_equivalent, x_1_equivalent).detach().cpu().numpy()

            # cosine of paired conditions
            cosineLoss = torch.nn.functional.cosine_similarity(z_latent_base_1_equivalent,z_latent_base_2_equivalent,dim=-1).mean()
            print2log('cosine of trained paired conditions: {:.4f}'.format(cosineLoss.item()))
        else:
            pearson_1_to_2 = np.nan
            pearson_2_to_1 = np.nan

    return (pearson_1_to_2, pearson_2_to_1,pear_1,pear_2,decoder_1, decoder_2, encoder_1, encoder_2, adverse_classifier) #pear_2,f1,class_acc and encoder_2, classifier, adverse_classifier, Vsp

def train_and_match(model_params, device, x_1_train, x_2_train,
                    decoder_1, decoder_2, encoder_1, encoder_2, adverse_classifier, classifier, Vsp,
                    bs_1:int, bs_2:int, NUM_EPOCHS:int,
                    class_criterion,
                    unpaired_mixing_limit:int=300,
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
    sinkhornLoss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    
    # Training loop
    for e in range(NUM_EPOCHS):

        if (e==unpaired_mixing_limit) or (e >= unpaired_mixing_limit and e % 300 == 0):
            if e == unpaired_mixing_limit:
                print2log('Unpaired mixing limit reached!')
            print2log('Re-matching systems...')
            decoder_1.eval()
            encoder_2.eval()
            Vsp.eval()            
            with torch.no_grad():
                z_base_2 = encoder_2(x_2_train.double().to(device))
                z_species_2 = torch.cat((torch.zeros(x_2_train.shape[0], 1), torch.ones(x_2_train.shape[0], 1)), 1).double().to(device)
                z_2_translated = Vsp(z_base_2,1 - z_species_2)
                x2_translated = decoder_2(z_2_translated)
            matched_conditions = partitionAndMatch(x_1_train.detach().cpu().numpy(),
                                                   x2_translated.detach().cpu().numpy(),
                                                   list(np.arange(x_1_train.shape[0])),
                                                   list(np.arange(x2_translated.shape[0])))
            # matched_conditions.to_csv('matched_conditions'+str(e)+'.csv')
            # Ensure all samples are included in the binary matrices
            all_system1_samples = np.arange(x_1_train.shape[0])
            all_system2_samples = np.arange(x2_translated.shape[0])
            # Create binary matrix matching all system 1 with all system 2
            conditions12 = matched_conditions.loc[:, ['system1_samples', 'system2_samples']]
            conditions12['value'] = 1
            # Remove duplicates to ensure unique combinations
            conditions12 = conditions12.drop_duplicates()
            conditions12 = conditions12.pivot(index='system1_samples', columns='system2_samples', values='value')
            conditions12 = conditions12.reindex(index=all_system1_samples, columns=all_system2_samples, fill_value=0)
            conditions12 = conditions12.fillna(0)
            conditions12 = conditions12.values
            # Create binary matrix matching all system 1 with each other
            conditions1 = matched_conditions.loc[:, ['system1_samples', 'system1_label']].drop_duplicates()
            conditions1.set_index('system1_samples', inplace=True)
            conditions1 = 1 * (conditions1.values == conditions1.values.T)
            conditions1 = pd.DataFrame(conditions1, 
                                       index=matched_conditions.loc[:, ['system1_samples', 'system1_label']].drop_duplicates().system1_samples, 
                                       columns=matched_conditions.loc[:, ['system1_samples', 'system1_label']].drop_duplicates().system1_samples)
            conditions1 = conditions1.reindex(index=all_system1_samples, columns=all_system1_samples, fill_value=0)
            conditions1 = conditions1.values
            # Create binary matrix matching all system 2 with each other
            conditions2 = matched_conditions.loc[:, ['system2_samples', 'system2_label']].drop_duplicates()
            conditions2.set_index('system2_samples', inplace=True)
            conditions2 = 1 * (conditions2.values == conditions2.values.T)
            conditions2 = pd.DataFrame(conditions2, 
                                       index=matched_conditions.loc[:, ['system2_samples', 'system2_label']].drop_duplicates().system2_samples, 
                                       columns=matched_conditions.loc[:, ['system2_samples', 'system2_label']].drop_duplicates().system2_samples)
            conditions2 = conditions2.reindex(index=all_system2_samples, columns=all_system2_samples, fill_value=0)
            conditions2 = conditions2.values
        
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
            trainloader_2 += getSamples(N_2, bs_2)[:maxLen - len(trainloader_2)]
            while len(trainloader_2) < maxLen:
                trainloader_2 += getSamples(N_2, bs_2)[:maxLen - len(trainloader_2)]
        
        # Iterate through batches
        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]
            
            X_1 = x_1_train[dataIndex_1, :].double().to(device)
            X_2 = x_2_train[dataIndex_2, :].double().to(device)

            
            if e>=unpaired_mixing_limit:
                c1 = np.concatenate((conditions1[dataIndex_1,:][:,dataIndex_1],conditions12[dataIndex_1,:][:,dataIndex_2]),axis=1)
                c2 = np.concatenate((conditions12[dataIndex_1,:][:,dataIndex_2].T,conditions2[dataIndex_2,:][:,dataIndex_2]),axis=1)
                conditions = np.concatenate((c1,c2),axis=0)
                mask = torch.tensor(conditions).double().to(device).detach()

            
            # Species vectors
            z_species_1 = torch.cat((torch.ones(X_1.shape[0], 1), torch.zeros(X_1.shape[0], 1)), 1).double().to(device)
            z_species_2 = torch.cat((torch.zeros(X_2.shape[0], 1), torch.ones(X_2.shape[0], 1)), 1).double().to(device)
            
            optimizer.zero_grad()
            optimizer_adv.zero_grad()
                        
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

            # now perform the non-aversesary step
            optimizer_adv.step()
            optimizer.zero_grad()
            z_base_1 = encoder_1(X_1)
            z_base_2 = encoder_2(X_2)
            latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)
            
            z_1 = Vsp(z_base_1,z_species_1)
            z_2 = Vsp(z_base_2,z_species_2)
            latent_vectors = torch.cat((z_1, z_2), 0)
            
            y_pred_1 = decoder_1(z_1)
            fitLoss_1 = torch.mean(torch.sum((y_pred_1 - X_1)**2,dim=1))
            L2Loss_1 = decoder_1.L2Regularization(model_params['dec_l2_reg']) + encoder_1.L2Regularization(model_params['enc_l2_reg'])
            loss_1 = fitLoss_1 + L2Loss_1
            
            y_pred_2 = decoder_2(z_2)
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

            ## Similar conditions should be close in the basal latent space
            if e < unpaired_mixing_limit:
                ## minimize Wasserstein distance
                dist_loss = sinkhornLoss(z_base_1,z_base_2) # approximation of wasserstein distance
                dist_loss = dist_loss.mean()
            else:
                silimalityLoss = torch.sum(
                torch.cdist(latent_base_vectors, latent_base_vectors) * mask.float()) / mask.float().sum()
                w1 = latent_base_vectors.norm(p=2, dim=1, keepdim=True)
                w2 = latent_base_vectors.norm(p=2, dim=1, keepdim=True)
                cosineLoss = torch.mm(latent_base_vectors, latent_base_vectors.t()) / (w1 * w2.t()).clamp(min=1e-6)
                cosineLoss = torch.sum(cosineLoss * mask.float()) / mask.float().sum()
                dist_loss = silimalityLoss - 10.0 * cosineLoss
            
            loss = loss_1 + loss_2 +model_params['reg_classifier'] * entropy + model_params['distance_reg']*dist_loss - model_params['reg_adv']*adv_entropy +classifier.L2Regularization(model_params['state_class_reg']) +Vsp.Regularization(model_params['v_reg'])
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
        if (e>=0):
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
            outString += ',distance loss={:.4f}'.format(dist_loss.item())

        # Logging
        if (e % 250 == 0 and e > 0) or (e == 0) or (e + 1 == NUM_EPOCHS):
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
        pear_1 = pearson_r(y_pred_1.detach(), x_1_train.double().to(device)).cpu().numpy()
        pear_2 = pearson_r(y_pred_2.detach(), x_2_train.double().to(device)).cpu().numpy()
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
            pearson_1_to_2 = pearson_r(x_hat_2_equivalent, x_2_equivalent).detach().cpu().numpy()
            # second translate system 2 to 1
            z_latent_base_2_equivalent  = encoder_2(x_2_equivalent)
            z_latent_2_equivalent = Vsp(z_latent_base_2_equivalent,z_species_2_equivalent)
            x_hat_1_equivalent = decoder_1(z_latent_2_equivalent).detach()
            pearson_2_to_1 = pearson_r(x_hat_1_equivalent, x_1_equivalent).detach().cpu().numpy()
        else:
            pearson_1_to_2 = np.nan
            pearson_2_to_1 = np.nan

    return (pearson_1_to_2, pearson_2_to_1,pear_1,pear_2,f1,class_acc,decoder_1, decoder_2, encoder_1, encoder_2, classifier, adverse_classifier, Vsp)


def train_autotransop(model_params, device,alldata, trainInfo_1, trainInfo_2,trainInfo_paired,
                    decoder_1, decoder_2, encoder_1, encoder_2, adverse_classifier, classifier, Vsp,prior_d, local_d,
                    bs_1:int, bs_2:int,bs_paired:int, NUM_EPOCHS:int,
                    class_criterion,
                    pairs_train=None):
    
    # make the whole train data
    x_1_train = torch.tensor(np.concatenate((alldata.loc[trainInfo_paired['sig_id.x']].values,
                                             alldata.loc[trainInfo_1.sig_id].values))).float().to(device)
    x_2_train = torch.tensor(np.concatenate((alldata.loc[trainInfo_paired['sig_id.y']].values,
                                       alldata.loc[trainInfo_2.sig_id].values))).float().to(device)
    # Combine parameters and create optimizers/schedulers
    allParams = list(decoder_1.parameters()) + list(decoder_2.parameters()) + \
                list(encoder_1.parameters()) + list(encoder_2.parameters()) + \
                list(classifier.parameters()) + list(Vsp.parameters()) + \
                list(prior_d.parameters()) + list(local_d.parameters())
    
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
    N_paired = len(trainInfo_paired)
    N_1 = len(trainInfo_1)
    N_2 = len(trainInfo_2)
    N = N_1
    if N_2>N:
        N=N_2

    # ## distance criterion
    # sinkhornLoss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    
    # Training loop
    for e in range(NUM_EPOCHS):

        decoder_1.train()
        decoder_2.train()
        encoder_1.train()
        encoder_2.train()
        classifier.train()
        adverse_classifier.train()
        Vsp.train()
        prior_d.train()
        local_d.train()

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
        for j in range(maxLen):
            dataIndex_1 = trainloader_1[j]
            dataIndex_2 = trainloader_2[j]
            dataIndex_paired = trainloader_paired[j]

            # Get batch
            df_pairs = trainInfo_paired.iloc[dataIndex_paired,:]
            df_1 = trainInfo_1.iloc[dataIndex_1,:]
            df_2 = trainInfo_2.iloc[dataIndex_2,:]
            paired_inds = len(df_pairs)
            
            X_1 = torch.tensor(np.concatenate((alldata.loc[df_pairs['sig_id.x']].values,
                                                 alldata.loc[df_1.sig_id].values))).double().to(device)
            X_2 = torch.tensor(np.concatenate((alldata.loc[df_pairs['sig_id.y']].values,
                                               alldata.loc[df_2.sig_id].values))).double().to(device)
            
            conditions = np.concatenate((df_pairs.conditionId.values,
                                            df_1.conditionId.values,
                                            df_pairs.conditionId.values,
                                            df_2.conditionId.values))
            size = conditions.size
            conditions = conditions.reshape(size,1)
            conditions = conditions == conditions.transpose()
            conditions = conditions*1
            mask = torch.tensor(conditions).to(device).detach()
            pos_mask = mask
            neg_mask = 1 - mask
            log_2 = math.log(2.)

            # Species vectors
            z_species_1 = torch.cat((torch.ones(X_1.shape[0], 1), torch.zeros(X_1.shape[0], 1)), 1).double().to(device)
            z_species_2 = torch.cat((torch.zeros(X_2.shape[0], 1), torch.ones(X_2.shape[0], 1)), 1).double().to(device)
            
            optimizer.zero_grad()
            optimizer_adv.zero_grad()
                        
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

            # now perform the non-aversesary step
            optimizer_adv.step()
            optimizer.zero_grad()
            z_base_1 = encoder_1(X_1)
            z_base_2 = encoder_2(X_2)
            latent_base_vectors = torch.cat((z_base_1, z_base_2), 0)

            # Compute mutual information terms
            z_un = local_d(latent_base_vectors)
            res_un = torch.matmul(z_un, z_un.t())
            p_samples = res_un * pos_mask.float()
            q_samples = res_un * neg_mask.float()
            Ep = log_2 - torch.nn.functional.softplus(- p_samples)
            Eq = torch.nn.functional.softplus(-q_samples) + q_samples - log_2
            Ep = (Ep * pos_mask.float()).sum() / pos_mask.float().sum()
            Eq = (Eq * neg_mask.float()).sum() / neg_mask.float().sum()
            mi_loss = Eq - Ep
            prior = torch.rand_like(latent_base_vectors)
            term_a = torch.log(prior_d(prior)).mean()
            term_b = torch.log(1.0 - prior_d(latent_base_vectors)).mean()
            prior_loss = -(term_a + term_b) * model_params['prior_beta']
            
            z_1 = Vsp(z_base_1,z_species_1)
            z_2 = Vsp(z_base_2,z_species_2)
            latent_vectors = torch.cat((z_1, z_2), 0)
            
            y_pred_1 = decoder_1(z_1)
            fitLoss_1 = torch.mean(torch.sum((y_pred_1 - X_1)**2,dim=1))
            L2Loss_1 = decoder_1.L2Regularization(model_params['dec_l2_reg']) + encoder_1.L2Regularization(model_params['enc_l2_reg'])
            loss_1 = fitLoss_1 + L2Loss_1
            
            y_pred_2 = decoder_2(z_2)
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

            ## Similar conditions should be close in the basal latent space
            silimalityLoss = torch.mean(torch.sum((z_base_1[0:paired_inds,:] - z_base_2[0:paired_inds,:])**2,dim=-1))
            cosineLoss = torch.nn.functional.cosine_similarity(z_base_1[0:paired_inds,:],z_base_2[0:paired_inds,:],dim=-1).mean()
            # silimalityLoss = torch.sum(torch.cdist(latent_base_vectors, latent_base_vectors) * mask.float()) / mask.float().sum()
            # w1 = latent_base_vectors.norm(p=2, dim=1, keepdim=True)
            # w2 = latent_base_vectors.norm(p=2, dim=1, keepdim=True)
            # cosineLoss = torch.mm(latent_base_vectors, latent_base_vectors.t()) / (w1 * w2.t()).clamp(min=1e-6)
            # cosineLoss = torch.sum(cosineLoss * mask.float()) / mask.float().sum()
            
            dist_loss = silimalityLoss - cosineLoss
            
            loss = loss_1 + loss_2 +model_params['reg_classifier'] * entropy + model_params['distance_reg']*dist_loss - model_params['reg_adv']*adv_entropy +classifier.L2Regularization(model_params['state_class_reg']) +Vsp.Regularization(model_params['v_reg'])+model_params['lambda_mi_loss']*mi_loss + prior_loss
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
        if (e>=0):
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
            # outString += ',distance loss={:.4f}'.format(dist_loss.item())
            outString += ',MI={:.4f}'.format(mi_loss.item())
            outString += ',prior loss={:.4f}'.format(prior_loss.item())
            outString += ',similarity loss={:.4f}'.format(silimalityLoss.item())
            outString += ',cosine loss={:.4f}'.format(cosineLoss.item())

        # Logging
        if (e % 250 == 0 and e > 0) or (e == 0) or (e + 1 == NUM_EPOCHS):
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
        pear_1 = pearson_r(y_pred_1.detach(), x_1_train.double().to(device)).cpu().numpy()
        pear_2 = pearson_r(y_pred_2.detach(), x_2_train.double().to(device)).cpu().numpy()
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
            pearson_1_to_2 = pearson_r(x_hat_2_equivalent, x_2_equivalent).detach().cpu().numpy()
            # second translate system 2 to 1
            z_latent_base_2_equivalent  = encoder_2(x_2_equivalent)
            z_latent_2_equivalent = Vsp(z_latent_base_2_equivalent,1.-z_species_2_equivalent)
            x_hat_1_equivalent = decoder_1(z_latent_2_equivalent).detach()
            pearson_2_to_1 = pearson_r(x_hat_1_equivalent, x_1_equivalent).detach().cpu().numpy()
        else:
            pearson_1_to_2 = np.nan
            pearson_2_to_1 = np.nan

    return (pearson_1_to_2, pearson_2_to_1,pear_1,pear_2,f1,class_acc,decoder_1, decoder_2, encoder_1, encoder_2, classifier, adverse_classifier, Vsp, prior_d, local_d)

