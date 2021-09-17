import os
import json

import examples.datahandlers as datahandlers
from ibmfl.util.datasets import load_adult
from ibmfl.util.datasets import load_nursery
from ibmfl.util.data_handlers.adult_dt_data_handler import AdultDTDataHandler


def get_fusion_config():
    fusion = {
        'name': 'ID3FusionHandler',
        'path': 'ibmfl.aggregator.fusion.dt_fusion_handler'
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
            'max_depth': 3,
        }
    }

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model=None):

    SUPPORTED_DATASETS = ['adult', 'nursery', 'custom_dataset']
    if dataset in SUPPORTED_DATASETS:
        data = datahandlers.get_datahandler_config(
            dataset, folder_data, party_id, is_agg)
    else:
        raise Exception(
            "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model=None):

    if dataset == 'adult':
        loaded_data = load_adult()
        # preproces the dataset first before generating the model spec
        dh = AdultDTDataHandler()
        loaded_data = dh.preprocess(loaded_data)
    elif dataset == 'nursery':
        loaded_data = load_nursery()
    spec = dict()
    spec['list_of_features'] = list(range(loaded_data.shape[1] - 1))

    feature_values = list()
    for feature in range(loaded_data.shape[1]):
        if loaded_data.columns[feature] != 'class':
            new_feature = loaded_data[loaded_data.columns[feature]
                                      ].cat.categories
            feature_values.append(new_feature.tolist())
    spec['feature_values'] = feature_values

    list_of_labels = loaded_data['class'].cat.categories
    spec['list_of_labels'] = list_of_labels.tolist()

    f_spec = os.path.join(folder_configs, 'dt_model_spec.json')
    with open(f_spec, 'w') as f:
        json.dump(spec, f)

    spec = {
        'model_name': 'decision-tree',
        'model_definition': os.path.join(folder_configs, 'dt_model_spec.json')
    }

    model = {
        'name': 'DTFLModel',
        'path': 'ibmfl.model.dt_fl_model',
        'spec': spec
    }

    return model
