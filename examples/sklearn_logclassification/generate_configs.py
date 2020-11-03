import os
import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier

import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {
        'name': 'IterAvgFusionHandler',
        'path': 'ibmfl.aggregator.fusion.iter_avg_fusion_handler'
    }
    return fusion


def get_local_training_config():
    local_training_handler = {
        'name': 'LocalTrainingHandler',
        'path': 'ibmfl.party.training.local_training_handler'
    }
    return local_training_handler


def get_hyperparams():
    hyperparams = {
        'global': {
            'rounds': 3,
            'termination_accuracy': 0.9
        },
        'local': {
            'training': {
                'max_iter': 2
            }
        }
    }

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False):

    SUPPORTED_DATASETS = ['adult', 'mnist']
    if dataset in SUPPORTED_DATASETS:
        if dataset == 'adult':
            dataset = 'adult_sklearn'
        elif dataset == 'mnist':
            dataset = 'mnist_sklearn'
        data = datahandlers.get_datahandler_config(
            dataset, folder_data, party_id, is_agg)
    else:
        raise Exception(
            "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):
    if is_agg:
        return None

    model = SGDClassifier(loss='log', penalty='l2')

    if dataset == 'adult':
        model.classes_ = np.array([0, 1])
    elif dataset == 'mnist':
        model.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    fname = os.path.join(folder_configs, 'model_architecture.pickle')

    with open(fname, 'wb') as f:
        pickle.dump(model, f)

    # Generate model spec:
    spec = {
        'model_definition': fname
    }

    model = {
        'name': 'SklearnSGDFLModel',
        'path': 'ibmfl.model.sklearn_SGD_linear_fl_model',
        'spec': spec
    }

    return model
