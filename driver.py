import faulthandler
faulthandler.enable()
import sys
import numpy as np
import random
import torch
import tqdm
import os
import logging

from global_configs import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


import h5py
import time
from collections import defaultdict, OrderedDict
import argparse
import pickle as pkl
import time
import json, os, ast, h5py
import math

#from models import MFN
from models import DeceptionModel


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from sacred import Experiment
from tqdm import tqdm


ex = Experiment('mental_face_transfer')
from sacred.observers import MongoObserver

#We must change url to the the bluehive node on which the mongo server is running
url_database = conf_url_database
mongo_database_name = conf_mongo_database_name


ex.observers.append(MongoObserver.create(url= url_database ,db_name= mongo_database_name))

my_logger = logging.getLogger()
my_logger.disabled=True

@ex.config
def cfg():
    node_index = 0
    epoch = 50 #paul did 50
    shuffle = True
    num_workers = 2
    best_model_path =  "/scratch/echowdh2/saved_models_from_projects/mental_face_transfer/"+str(node_index) +"_best_model.chkpt"
    experiment_config_index=0


    dataset_location = None
    dataset_name = None

    d_summary=None
    d_pretrained=None    
    sum_fc1_in = 94
    sum_fc1_out = random.choice([16,32,40,48,56,64])
    sum_fc1_drop = random.choice([0.1,0.2,0.3,0.4,0.5])
    sum_fc2_in = sum_fc1_out
    sum_fc2_out = 1 

    pre_fc1_in = d_pretrained
    pre_fc1_out = random.choice([2,4,6,8,16])
    pre_fc1_drop = random.choice([0.1,0.2,0.3,0.4,0.5])
    pre_fc2_in = pre_fc1_out
    pre_fc2_out = 1 


    padding_value = 0.0

    train_batch_size=140
    #putting everything at the same time, I believe it will be fine
    dev_batch_size= 36
    test_batch_size= 44

    use_summary=True
    use_pretrained=True

    loss_function="bce"
        
    save_model = True
    save_mode = 'best'

    prototype=None

    prot_train=10
    prot_dev=10
    prot_test=10
    prot_inference = 20

    if prototype:
        epoch=5
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = random.choice(np.logspace(-2,0,num=10))



class DeceptionDataset(Dataset):
    
    def __init__(self,X,y):
        
        self.X=X
        self.y=np.array(y,dtype=int)
        
    def __len__(self):
        return len(self.y)   
    
    def __getitem__(self,i):
        
        return torch.FloatTensor(self.X[i]),torch.FloatTensor([self.y[i]])


def get_data(root_list,openface_sdk,affectiva_sdk,relevant_pretrained_features,relevant_data_property):
    x,y=[],[]
    for root in root_list:
        of_segments=openface_sdk[root]['segment_openface_features']
        aff_segments=affectiva_sdk[root]['segment_affectiva_features']
        
        all_of=[]
        for segment in of_segments:
            for sub_segment in segment:
                all_of.append(sub_segment)
                
        of_mean=np.mean(all_of,axis=0)
        
        all_aff=[]
        for segment in aff_segments:
            for sub_segment in segment:
                all_aff.append(sub_segment)
                
        aff_mean=np.mean(all_aff,axis=0)
        
        pretrained=relevant_pretrained_features[root]['pretrained_probs']
        all_f=np.concatenate((of_mean,aff_mean))
        
        all_f=np.concatenate((all_f,pretrained))
        
        x.append(all_f.tolist())
        y.append(relevant_data_property[root]['truth_val'])
        
    return np.array(x),np.array(y)


@ex.capture
def set_up_data_loader(_config):
    
    base_path="/home/kamrul/Research/acii_2019/mental_faces/data/relevant/"
    relevant_time_sdk=pkl.load(open(base_path+"relevant_time_sdk.pkl","rb"))
    relevant_data_property=pkl.load(open(base_path+"relevant_data_property_sdk.pkl","rb"))
    relevant_pretrained_features=pkl.load(open(base_path+"pretrained_features_without_covarep_sdk.pkl","rb"))
        
    openface_file="/home/kamrul/Research/acii_2019/mental_faces/data/relevant/relevant_openface2_sdk.pkl"
    affectiva_file="/home/kamrul/Research/acii_2019/mental_faces/data/relevant/relevant_affectiva_sdk.pkl"
    covarep_file="/home/kamrul/Research/acii_2019/mental_faces/data/relevant/relevant_covarep_sdk.pkl"
    openface_sdk=pkl.load(open(openface_file,"rb"))
    affectiva_sdk=pkl.load(open(affectiva_file,"rb"))
    covarep_sdk=pkl.load(open(covarep_file,"rb"))
    
    train_roots=['08-33-112', '2018-02-25_15-05-13-792', '2018-08-12_12-56-37-196', '2018-02-25_17-20-28-378', '2018-02-17_16-12-17-891', '29-09-99', '2018-08-12_12-36-33-550', '38-26-565', '13-20-248', '2018-02-25_17-05-45-845', '21-16-706', '12-23-235', '2018-08-12_13-53-02-58', '05-06-501', '16-19-112', '48-39-430', '2018-03-04_18-21-30-606', '11-22-452', '01-50-492', '19-27-616', '2018-02-17_14-45-40-200', '58-57-956', '06-44-152', '00-34-288', '31-23-340', '19-04-926', '2018-09-01_13-15-43-356', '2018-02-25_13-49-51-506', '04-57-574', '44-30-30', '30-57-338', '2018-02-25_17-20-56-708', '2018-03-04_17-12-05-975', '2018-09-08_12-01-43-738', '2018-02-25_17-51-49-71', '2018-02-25_17-33-34-929', '2018-06-08_15-49-48-161', '2018-09-08_12-09-26-1', '11-22-778', '2018-03-04_17-25-20-50', '2018-02-25_18-05-08-996', '2018-02-25_18-06-25-166', '2018-08-12_14-03-49-927', '2018-06-08_14-27-12-890', '2018-02-25_16-53-19-560', '48-00-366', '39-49-946', '2018-02-17_15-30-27-188', '11-09-515', '2018-02-25_14-07-04-471', '2018-03-04_17-37-48-342', '48-02-966', '30-42-250', '22-31-901', '16-40-42', '2018-03-04_18-34-39-751', '37-36-401', '04-38-645', '27-51-501', '2018-08-12_12-05-51-37', '19-35-145', '2018-06-08_14-09-16-969', '2018-02-25_16-20-27-786', '28-32-985', '15-21-96', '2018-02-25_16-35-15-258', '2018-02-25_13-33-09-475', '2018-02-25_14-21-26-22', '2018-06-08_15-09-33-44', '15-58-679', '2018-03-03_15-11-18-614', '22-41-332', '2018-06-08_13-39-21-846', '2018-09-01_17-08-02-6', '18-43-166', '15-29-238', '2018-03-04_17-11-56-584', '12-03-237', '27-35-915', '2018-08-12_13-37-20-143', '10-48-352', '2018-03-03_14-50-02-899', '2018-02-25_14-35-41-678', '2018-09-01_16-41-44-747', '2018-02-25_17-47-15-16', '2018-02-25_16-06-52-924', '28-52-283', '16-56-369', '2018-06-08_15-51-08-556', '2018-03-03_14-55-34-435', '33-47-982', '36-53-693', '2018-03-03_16-37-02-627', '24-43-478', '20-28-731', '2018-06-08_15-22-36-67', '15-13-468', '2018-08-12_11-49-35-321', '03-23-90', '05-50-814', '2018-03-04_17-52-53-448', '18-48-414', '17-15-163', '2018-03-04_16-45-46-434', '2018-02-25_15-05-31-847', '2018-02-25_14-21-51-412', '2018-06-08_15-35-18-558', '33-07-644', '12-07-438', '22-10-468', '13-27-771', '20-40-167', '09-02-798', '2018-09-08_13-23-19-683', '39-18-93', '11-44-333', '38-29-906', '2018-02-17_15-15-12-644', '03-42-701', '21-24-649', '12-38-74', '2018-03-04_16-23-14-328', '2018-09-08_12-34-07-349', '2018-08-12_12-58-02-439', '11-59-155', '2018-02-25_14-49-36-200', '2018-09-08_12-34-17-60', '20-38-706', '05-15-634', '40-18-142', '2018-08-12_14-06-23-202', '11-26-504', '2018-02-17_14-33-35-477', '2018-02-25_13-14-22-215', '36-04-870', '2018-09-01_12-35-37-636', '12-05-243', '19-05-68', '2018-09-08_11-43-44-665', '2018-09-08_13-09-49-380', '21-03-94', '14-08-617', '11-45-95', '2018-03-03_14-31-25-188', '03-12-820', '14-08-252', '2018-09-08_12-17-43-362', '18-39-581', '35-42-851', '2018-09-01_15-26-01-472', '2018-06-08_13-58-18-9', '27-19-148', '14-04-955', '26-17-966', '2018-02-25_17-06-48-147', '2018-02-25_13-34-11-253', '40-07-485', '05-49-922', '05-44-437', '28-32-854', '2018-09-08_11-26-50-995', '2018-02-25_16-05-37-615', '2018-03-03_16-23-45-184', '29-48-550', '2018-09-08_13-37-59-49', '2018-03-04_17-40-24-748', '04-30-467', '11-17-124', '2018-09-01_11-33-29-289', '01-01-49', '2018-09-08_11-56-25-567', '50-50-13', '09-14-629', '2018-09-01_11-20-44-762', '37-32-755', '26-15-622']
    test_roots=['36-37-921', '2018-02-17_15-59-37-779', '2018-03-03_16-08-55-856', '2018-08-12_14-21-17-501', '2018-06-08_14-44-26-144', '22-35-549', '24-30-926', '2018-02-25_14-05-58-190', '2018-03-04_18-06-42-851', '20-05-755', '2018-08-12_11-31-52-459', '2018-08-12_11-51-07-800', '06-50-611', '09-32-917', '2018-09-08_11-27-01-675', '02-09-31', '2018-06-08_15-35-59-88', '2018-09-01_15-42-41-661', '2018-06-08_14-59-15-472', '2018-03-03_16-49-28-573', '06-57-279', '2018-02-17_14-58-58-929', '03-47-890', '09-18-253', '2018-03-03_15-55-22-859', '2018-03-04_16-58-01-813', '13-01-549', '2018-03-04_18-20-44-311', '2018-06-08_13-38-05-898', '2018-08-12_14-21-19-68', '2018-09-08_12-54-50-681', '2018-02-25_15-51-57-535', '16-48-797', '2018-03-03_15-40-45-399', '54-10-129', '2018-03-04_17-52-36-829', '28-45-33', '2018-03-03_15-54-55-137', '46-30-695', '2018-06-08_15-22-30-837', '2018-02-25_13-50-22-285', '24-24-724', '18-42-263', '2018-09-08_13-22-56-739']
    
    all_train_x,all_train_y=get_data(train_roots)
    test_x,test_y=get_data(test_roots)
    sorted_index=np.argsort(all_train_y)
    dev_index=sorted_index[0:18].tolist()+sorted_index[-18:].tolist()
    train_index=sorted_index[18:-18].tolist()
    np.random.shuffle(dev_index)
    np.random.shuffle(train_index)
    dev_x=all_train_x[dev_index]
    dev_y=all_train_y[dev_index]
    train_x=all_train_x[train_index]
    train_y=all_train_y[train_index]

    train_data=DeceptionDataset(train_x,train_y)
    dev_data=DeceptionDataset(dev_x,dev_y)
    test_data=DeceptionDataset(test_x,test_y)

    train_dataloader = DataLoader(train_data, batch_size=len(train_x), shuffle=True)
    dev_dataloader = DataLoader(dev_data, batch_size=len(dev_x), shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_x), shuffle=True)


    return train_dataloader,dev_dataloader,test_dataloader


@ex.capture
def set_random_seed(_seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    Parameter:
        seed: It is set in @ex.config and will be passed through variable injection.
    """
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)



@ex.capture
def train_epoch(model, training_data, criterion,optimizer, device,_config):
    ''' Epoch operation in training phase'''

    model.train()

    epoch_loss = 0.0
    num_batches = 0
    #sDismissed:tep_and_update_lr() is being called on every epoch. May be, it was fine when the batch size was very big. Now,the batch size is 16.
    #So, we may take one update every 30 iterations since it seems that 16 will be highest batch size and 16*30=480 which is a regular batch size
    #Or since there are 1200 data and 75 batches each with 16 data, we can take an update after every 25 batch.
   
   
    for batch in tqdm(training_data, mininterval=2,desc='  - (Training)   ', leave=False):

     #TODO: For simplicity, we are not using X_pos right now as we really do not know
     #how it can be used properly. So, we will just use the context information only.
        #X_Punchline,X_Context,X_pos_Context,Y = map(lambda x: x.to(device), batch)
        X,y = map(lambda x: x.to(device).float(), batch)
        
      
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y.squeeze(1))
        
        loss.backward()
        #optimizer.step()
        epoch_loss += loss.item()
        
        optimizer.step()


        num_batches +=1

   
    return epoch_loss / num_batches


@ex.capture
def eval_epoch(model,data_loader,criterion, device,_config):
    ''' Epoch operation in evaluation phase '''
    epoch_loss = 0.0
    num_batches=0
    model.eval()
    with torch.no_grad():

        for batch in tqdm(data_loader, mininterval=2,desc='  - (Validation)   ', leave=False):
            
            X,y = map(lambda x: x.to(device).float(), batch)
            #print("Dev:",X)

            predictions = model(X)
            loss = criterion(predictions, y.squeeze(1))
            
            epoch_loss += loss.item()
            
            num_batches +=1

    return epoch_loss / num_batches


@ex.capture
def reload_model_from_file(file_path):
    checkpoint = torch.load(file_path)
    _config = checkpoint['_config']
    
    #encoder_config = _config["multimodal_context_configs"]
    model = Baseline(_config).to(_config["device"])
    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')

    return model


@ex.capture
def test_epoch(model,data_loader,criterion, device,_config):
    ''' Epoch operation in evaluation phase '''
    epoch_loss = 0.0
    num_batches=0
    model.eval()
    returned_Y = None
    returned_predictions = None
    with torch.no_grad():
   
        for batch in tqdm(data_loader, mininterval=2,desc='  - (Test)   ', leave=False):
    
         
            X,y = map(lambda x: x.to(device).float(), batch)
            #print("Dev:",X)

            predictions = model(X)
            loss = criterion(predictions, y.squeeze(1))
            
            epoch_loss += loss.item()
            
            num_batches +=1
            #if we don'e do the squeeze, it remains as 2d numpy arraya nd hence
            #creates problems like nan while computing various statistics on them
            temp_Y = y.squeeze(1).cpu().numpy()
            returned_Y = temp_Y if (returned_Y is None) else np.concatenate((returned_Y,temp_Y))

            temp_pred = predictions.cpu().data.numpy()
            returned_predictions = temp_pred if returned_predictions is None else np.concatenate((returned_predictions,temp_pred))
            
    return returned_predictions,returned_Y



@ex.capture
def train(model, training_data, validation_data, optimizer,criterion,_config,_run):
    ''' Start training '''
    model_path = _config["best_model_path"]

    valid_losses = []
    for epoch_i in range(_config["epoch"]):
        
        train_loss = train_epoch(
            model, training_data, criterion,optimizer, device = _config["device"])
        _run.log_scalar("training.loss", train_loss, epoch_i)
        


        valid_loss = eval_epoch(model, validation_data, criterion,device=_config["device"])
        _run.log_scalar("dev.loss", valid_loss, epoch_i)


        
        
        valid_losses.append(valid_loss)
        print("\nepoch:{},train_loss:{}, valid_loss:{}".format(epoch_i,train_loss,valid_loss))
      #Due to space3 constraint, we are not saving the models. There should be enough info
      #in sacred to reproduce the results on the fly
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            '_config': _config,
            'epoch': epoch_i}

        if _config["save_model"]:
            if _config["save_mode"] == 'best':
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_path)
                    print('    - [Info] The checkpoint file has been updated.')
    #After the entire training is over, save the best model as artifact in the mongodb, only if it is not protptype
    #Due to space constraint, we are not saving the model since it is not necessary as we know the seed. If we need to regenrate the result
    #simple running it again should work
    # if(_config["prototype"]==False):
    #     ex.add_artifact(model_path)


@ex.capture
def test_score_from_file(test_data_loader,criterion,_config,_run):
    model_path =  _config["best_model_path"]
    model = reload_model_from_file(model_path)

    predictions,y_test = test_epoch(model,test_data_loader,criterion,_config["device"])
    predicted_label = torch.round(torch.sigmoid(predictions))

    true_label=y_test
    f_score = round(f1_score(np.round(true_label),np.round(predicted_label),average='weighted'),5)
    ex.log_scalar("test.f_score",f_score)

    #print("Confusion Matrix :")
    confusion_matrix_result = confusion_matrix(true_label, predicted_label)
    #print(confusion_matrix_result)
    
    #print("Classification Report :")
    classification_report_score = classification_report(true_label, predicted_label, digits=5)
    #print(classification_report_score)
    
    accuracy = accuracy_score(true_label, predicted_label)
    print("Accuracy:",accuracy )
    
    _run.info['final_result']={'accuracy':accuracy,
             "f_score":f_score,"Confusion Matrix":confusion_matrix_result}
    return accuracy


@ex.capture
def prepare_for_training(_config):
    train_data_loader,dev_data_loader,test_data_loader = set_up_data_loader()
    
    model = DeceptionModel(_config).to(_config["device"])
        
    optimizer = optim.SGD(model.parameters(), lr=_config["lr"])
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(_config["device"])
    
    return train_data_loader,dev_data_loader,test_data_loader,model,optimizer,criterion



@ex.automain
def driver(_config,_run):
    #print(np.logspace(-5,0))
    set_random_seed()
    train_data_loader,dev_data_loader,test_data_loader,model,optimizer,criterion = prepare_for_training()
    train(model, train_data_loader,dev_data_loader, optimizer, criterion)
    #assert False



    #test_accuracy =  test_score_from_model(model,test_data_loader,criterion)
    
    test_accuracy = test_score_from_file(test_data_loader,criterion)
    ex.log_scalar("test.accuracy",test_accuracy)
    results = dict()
    #I believe that it will try to minimize the rest. Let's see how it plays out
    results["optimization_target"] = 1 - test_accuracy

    return results