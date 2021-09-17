import os
import pickle

from diffprivlib.models import GaussianNB

import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {
        'name': 'NaiveBayesFusionHandler',
        'path': 'ibmfl.aggregator.fusion.naive_bayes_fusion_handler'
    }

    return fusion


def get_local_training_config(configs_folder=None):
    local_training_handler = {
        'name': 'LocalTrainingHandler',
        'path': 'ibmfl.party.training.local_training_handler'
    }
    return local_training_handler


def get_hyperparams(model=None):
    hyperparams = {
        'global': {
        }
    }

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model='sklearn'):

    SUPPORTED_DATASETS = ['adult']
    if dataset in SUPPORTED_DATASETS:
        if dataset == 'adult':
            dataset = 'adult_sklearn'
        data = datahandlers.get_datahandler_config(
            dataset, folder_data, party_id, is_agg)
    else:
        raise Exception(
            "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model='sklearn'):
    if is_agg:
        return None

    model = GaussianNB(epsilon=1, bounds=([0] * 18, [1] * 18))

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
        'name': 'NaiveBayesFLModel',
        'path': 'ibmfl.model.naive_bayes_fl_model',
        'spec': spec
    }

    return model
