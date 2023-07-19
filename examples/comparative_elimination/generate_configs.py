import os
from importlib import import_module

import numpy as np

import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {
        "name": "ComparativeEliminationFusionHandler",
        "path": "ibmfl.aggregator.fusion.comparative_elimination_fusion_handler",
    }

    return fusion


def get_local_training_config(configs_folder=None):
    local_training_handler = {"name": "LocalTrainingHandler", "path": "ibmfl.party.training.local_training_handler"}
    return local_training_handler


def get_hyperparams(model):
    hyperparams = {"global": {"rounds": 3, "byzantine_threshold": 1, "max_timeout": 60}}
    current_module = globals().get("__package__")

    model_module = import_module("{}.model_{}".format(current_module, model))
    local_params_method = getattr(model_module, "get_hyperparams")

    local_params = local_params_method()
    hyperparams["local"] = local_params

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model="keras"):
    SUPPORTED_DATASETS = ["mnist", "adult", "custom_dataset"]
    if dataset in SUPPORTED_DATASETS:
        if model not in "keras":
            dataset = dataset + "_" + model

        data = datahandlers.get_datahandler_config(dataset, folder_data, party_id, is_agg)
    else:
        raise Exception("The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model="keras"):
    SUPPORTED_MODELS = ["keras", "pytorch", "tf"]

    if model not in SUPPORTED_MODELS:
        raise Exception("Invalid model config for this fusion algorithm")

    current_module = globals().get("__package__")

    model_module = import_module("{}.model_{}".format(current_module, model))
    method = getattr(model_module, "get_model_config")
    # is_agg is set to False similar to gradient_agg, since agg needs initial model
    return method(folder_configs, dataset, is_agg=False, party_id=0)
