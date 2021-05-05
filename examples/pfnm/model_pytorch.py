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
    model = nn.Sequential(nn.Flatten(1, -1),
                          nn.Linear(784, 256),
                          nn.ReLU(),
                          nn.Linear(256, 100),
                          nn.ReLU(),
                          nn.Linear(100, 50),
                          nn.ReLU(),
                          nn.Linear(50, 10),
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
