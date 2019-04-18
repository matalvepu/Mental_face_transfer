import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau

import h5py
import time
from collections import defaultdict, OrderedDict
import argparse
import pickle as pickle
import time
import json, os, ast, h5py


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class DeceptionModel(nn.Module):
    def __init__(self, _config):
        
        super().__init__()
        self._config=_config
        
        self.sum_fc1 = nn.Linear(_config['sum_fc1_in'], _config['sum_fc1_out'])
        self.sum_fc2 = nn.Linear(_config['sum_fc2_in'], _config['sum_fc2_out'])
        self.sum_fc1_drop = nn.Dropout(_config["sum_fc1_drop"])
        self.sum_scalar = Variable(torch.rand(1),requires_grad=True)
        
        self.pre_fc1 = nn.Linear(_config['pre_fc1_in'], _config['pre_fc1_out'])
        self.pre_fc2 = nn.Linear(_config['pre_fc2_in'], _config['pre_fc2_out'])
        self.pre_fc1_drop = nn.Dropout(_config["pre_fc1_drop"])
        self.pre_scalar = Variable(torch.rand(1),requires_grad=True)
        
    def forward(self, input_f):
        input_sum=input_f[:,0:86]
        input_pre=input_f[:,86:94]
        
        if self._config['use_summary']==False:
            input_sum=torch.zeros(input_sum.size()).to(self._config["device"])
            
        if self._config['use_pretrained']==False:
            input_pre=torch.zeros(input_pre.size()).to(self._config["device"])
        
        sum_prediction = self.sum_fc2(self.sum_fc1_drop(F.relu(self.sum_fc1(input_sum.squeeze(1))))).squeeze(1)
        pre_prediction = self.pre_fc2(self.pre_fc1_drop(F.relu(self.pre_fc1(input_pre.squeeze(1))))).squeeze(1)
        
        prediction = sum_prediction*self.sum_scalar.expand_as(sum_prediction)+pre_prediction*self.pre_scalar.expand_as(pre_prediction)
        return F.relu(prediction)