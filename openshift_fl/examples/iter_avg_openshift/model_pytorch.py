"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""
import os 
import torch
from torch import nn


def get_hyperparams():
    local_params = {
        'training': {
            'epochs': 3,
            'lr': 1
        },
        'optimizer': 'optim.Adadelta'
    }
    
    return local_params


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):

    if is_agg:
        return None
    model = nn.Sequential(nn.Conv2d(1, 32, 3, 1),
                          nn.ReLU(),
                          nn.Conv2d(32, 64, 3, 1),
                          nn.ReLU(),
                          nn.MaxPool2d(2, 2),
                          nn.Dropout2d(p=0.25),
                          nn.Flatten(),
                          nn.Linear(9216, 128),
                          nn.ReLU(),
                          nn.Dropout2d(p=0.5),
                          nn.Linear(128, 10),
                          nn.LogSoftmax(dim=1)
                          )
    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    # Save model
    fname = os.path.join(folder_configs, 'pytorch_sequence.pt')
    torch.save(model, fname)
    spec = {
        'model_name': 'pytorch-nn',
        'model_definition': fname
    }
    model = {
        'name': 'PytorchFLModel',
        'path': 'ibmfl.model.pytorch_fl_model',
        'spec': spec,
    }
    return model
