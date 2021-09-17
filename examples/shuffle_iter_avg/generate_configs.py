import os
import random
import sys
from importlib import import_module

import examples.datahandlers as datahandlers
_g_seed = None

def get_fusion_config():
    fusion = {
        'name': 'ShuffleIterAvgFusionHandler',
        'path': 'ibmfl.aggregator.fusion.shuffle_iter_avg_fusion_handler'
    }
    return fusion


def get_local_training_config(configs_folder=None):
    global _g_seed
    # default seed file
    seed_file = os.path.join(configs_folder, 'permute_secret.seed')

    if not _g_seed:
        _g_seed = random.randrange(sys.maxsize)
        with open(seed_file, 'w') as outfile:
            outfile.write(str(_g_seed))

    local_training_handler = {
        'name': 'ShuffleLocalTrainingHandler',
        'path': 'ibmfl.party.training.shuffle_local_training_handler',
        'info': {
            'permute_secret': seed_file
        }
    }
    return local_training_handler


def get_hyperparams(model):
    hyperparams = {
        'global': {
                'rounds': 3,
                'termination_accuracy': 0.9,
                'max_timeout': 60
            }
    }
    current_module = globals().get('__package__')

    model_module = import_module('{}.model_{}'.format(current_module, model))
    local_params_method = getattr(model_module, 'get_hyperparams')

    local_params = local_params_method()
    hyperparams['local'] = local_params
    
    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model='keras'):
    SUPPORTED_DATASETS = ['mnist', 'adult', 'cifar10', 'femnist', 'custom_dataset']
    if dataset in SUPPORTED_DATASETS:
        if model not in 'keras':
            dataset = dataset + "_" + model

        data = datahandlers.get_datahandler_config(
            dataset, folder_data, party_id, is_agg)
    else:
        raise Exception(
            "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model='keras'):
    SUPPORTED_MODELS = ['keras', 'pytorch', 'tf', 'sklearn']

    if model not in SUPPORTED_MODELS:
        raise Exception("Invalid model config for this fusion algorithm")

    current_module = globals().get('__package__')
    
    model_module = import_module('{}.model_{}'.format(current_module, model))
    method = getattr(model_module, 'get_model_config')

    return method(folder_configs, dataset, is_agg=is_agg, party_id=0)

