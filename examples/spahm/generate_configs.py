import os
import pickle
from importlib import import_module

from sklearn.cluster import KMeans

import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {"name": "SPAHMFusionHandler", "path": "ibmfl.aggregator.fusion.spahm_fusion_handler"}
    return fusion


def get_local_training_config(configs_folder=None):
    local_training_handler = {"name": "LocalTrainingHandler", "path": "ibmfl.party.training.local_training_handler"}
    return local_training_handler


def get_hyperparams(model="sklearn"):
    hyperparams = {
        "global": {"rounds": 1, "iters": 50, "optimize_hyperparams": True},
        "local": {"training": {"max_iter": 500, "n_clusters": 10}},
    }

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model="sklearn"):
    SUPPORTED_DATASETS = ["federated-clustering", "custom_dataset"]
    if dataset in SUPPORTED_DATASETS:
        data = datahandlers.get_datahandler_config(dataset, folder_data, party_id, is_agg)
    else:
        raise Exception("The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model="sklearn"):
    SUPPORTED_MODELS = ["sklearn"]

    if model not in SUPPORTED_MODELS:
        raise Exception("Invalid model config for this fusion algorithm")

    current_module = globals().get("__package__")

    model_module = import_module("{}.model_{}".format(current_module, model))
    method = getattr(model_module, "get_model_config")

    return method(folder_configs, dataset, is_agg=is_agg, party_id=0)
