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
        
        self.fc1 = nn.Linear(_config['sum_fc1_in'], _config['sum_fc1_out'])
        self.fc2 = nn.Linear(_config['sum_fc2_in'], _config['sum_fc2_out'])
        self.fc1_drop = nn.Dropout(_config["sum_fc1_drop"])
        
    def forward(self, input_f):
        input_f=torch.FloatTensor(input_f)
        prediction = self.fc2(self.fc1_drop(F.relu(self.fc1(input_f.squeeze(1))))).squeeze(1)
        
        return prediction