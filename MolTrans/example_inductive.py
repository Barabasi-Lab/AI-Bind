import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch import nn 
import copy

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc
from sklearn.model_selection import KFold
torch.manual_seed(1)    # reproducible torch:2 np:3
np.random.seed(1)

from sklearn.model_selection import train_test_split

from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

###################### Functions #########################################################################################################

def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())
        
        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()            
        
        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)
        
        loss_accumulate += loss
        count += 1
        
        logits = logits.detach().cpu().numpy()
        
        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
        
    loss = loss_accumulate/count
    
    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: "+ str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1=sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print ('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Sensitivity : ', sensitivity1 )

    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred, loss.item()


def main(fold_n, lr):
    config = BIN_config_DBPE()
    
    lr = lr
    BATCH_SIZE = config['batch_size']
    train_epoch = 100
    
    loss_history = []
    
    model = BIN_Interaction_Flat(**config)
    
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim = 0)
            
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    #opt = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    
    print('--- Data Preparation ---')
    
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6, 
              'drop_last': True}

    dataFolder = './dataset/BindingDB'
    df_train = pd.read_csv(dataFolder + '/train.csv')
    df_val = pd.read_csv(dataFolder + '/val.csv')
    df_test = pd.read_csv(dataFolder + '/test.csv')
    #df_train_val = pd.concat([df_train,df_val])

    import pdb; pdb.set_trace()

    ###### Make random split ######

    df_all = pd.concat([df_train,df_val,df_test])

    df_train, df_val_test = train_test_split(df_all, test_size=0.6) #(len(df_val)+len(df_test))/len(df_all))
    df_val, df_test = train_test_split(df_val_test, test_size=0.67) #len(df_test)/(len(df_val)+len(df_test)))
   
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    df_test = df_test.reset_index()
 
    df_train_val = pd.concat([df_train,df_val])

    ###############################

    train_val_smiles = list(set(df_train_val['SMILES'].tolist()))
    train_val_aas = list(set(df_train_val['Target Sequence'].tolist())) 

    ### Get the transductive test set ###

    df_test_transductive = df_test[df_test['SMILES'].isin(train_val_smiles) & df_test['Target Sequence'].isin(train_val_aas)]
    df_test_transductive = df_test_transductive.reset_index()
    df_test_transductive['Unnamed: 0'] = range(0,len(df_test_transductive))

    ### Get the semi-inductive test set ###

    df_test_semi_inductive = df_test[~df_test['SMILES'].isin(train_val_smiles) | ~df_test['Target Sequence'].isin(train_val_aas)]
    df_test_semi_inductive = df_test_semi_inductive.reset_index()
    df_test_semi_inductive['Unnamed: 0'] = range(0,len(df_test_semi_inductive))

    ### Get the inductive test set ###

    df_test_inductive = df_test[~df_test['SMILES'].isin(train_val_smiles) & ~df_test['Target Sequence'].isin(train_val_aas)]
    df_test_inductive = df_test_inductive.reset_index()
    df_test_inductive['Unnamed: 0'] = range(0,len(df_test_inductive))

    training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
    validation_generator = data.DataLoader(validation_set, **params)
    
    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)
 
    testing_set_transductive = BIN_Data_Encoder(df_test_transductive.index.values, df_test_transductive.Label.values, df_test_transductive)
    testing_generator_transductive = data.DataLoader(testing_set_transductive, **params)

    testing_set_semi_inductive = BIN_Data_Encoder(df_test_semi_inductive.index.values, df_test_semi_inductive.Label.values, df_test_semi_inductive)
    testing_generator_semi_inductive = data.DataLoader(testing_set_semi_inductive, **params)

    testing_set_inductive = BIN_Data_Encoder(df_test_inductive.index.values, df_test_inductive.Label.values, df_test_inductive)
    testing_generator_inductive = data.DataLoader(testing_set_inductive, **params)

    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)
    
    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(train_epoch):
        model.train()
        for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
            score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()
            
            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))
            
            loss = loss_fct(n, label)
            loss_history.append(loss)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if (i % 100 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(loss.cpu().detach().numpy()))
            
        # every epoch test
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss = test(validation_generator, model)
            if auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc
            
            print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: '+ str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1))
    
        print('--- Go for Testing ---')
        try:
            with torch.set_grad_enabled(False):
                auc, auprc, f1, logits, loss = test(testing_generator, model_max) 
                print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1) + ' , Test loss: '+str(loss))
                auc, auprc, f1, logits, loss = test(testing_generator_transductive, model_max)
                print('Transductive Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1) + ' , Test loss: '+str(loss))
                auc, auprc, f1, logits, loss = test(testing_generator_semi_inductive, model_max)
                print('Semi inductive Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1) + ' , Test loss: '+str(loss))
                auc, auprc, f1, logits, loss = test(testing_generator_inductive, model_max)
                print('Inductive Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1) + ' , Test loss: '+str(loss))
        except:
            print('testing failed')
    return model_max, loss_history


########################### Run MolTrans ##################################################################################################

# fold 1
#biosnap interaction times 1e-6, flat, batch size 64, len 205, channel 3, epoch 50
s = time()
model_max, loss_history = main(1, 5e-6)
e = time()
print(e-s)
lh = list(filter(lambda x: x < 1, loss_history))
#plt.plot(lh)

###########################################################################################################################################
