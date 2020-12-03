import os
import torch
from torch import nn

import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {
        'name': 'FedAvgFusionHandler',
        'path': 'ibmfl.aggregator.fusion.fedavg_fusion_handler'
    }
    return fusion


def get_local_training_config():
    local_training_handler = {
        'name': 'FedAvgLocalTrainingHandler',
        'path': 'ibmfl.party.training.fedavg_local_training_handler'
    }
    return local_training_handler


def get_hyperparams():
    hyperparams = {
        'global': {
            'rounds': 5,
            'termination_accuracy': 0.9
        },
        'local': {
            'training': {
                'epochs': 3,
                'lr': 1
            },
            'optimizer': 'optim.Adadelta'
        }
    }

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False):

    SUPPORTED_DATASETS = ['mnist']
    if dataset in SUPPORTED_DATASETS:
        if dataset == 'mnist':
            dataset = 'mnist_pytorch'
        data = datahandlers.get_datahandler_config(
            dataset, folder_data, party_id, is_agg)
    else:
        raise Exception(
            "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):
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
