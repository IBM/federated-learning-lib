from importlib import import_module

import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {"name": "Doc2VecFusionHandler", "path": "ibmfl.aggregator.fusion.doc2vec_fusion_handler"}
    return fusion


def get_local_training_config(configs_folder=None):
    local_training_handler = {
        "name": "Doc2VecLocalTrainingHandler",
        "path": "ibmfl.party.training.doc2vec_local_training_handler",
    }
    return local_training_handler


def get_hyperparams(model):
    hyperparams = {
        "global": {"rounds": 15, "max_timeout": 60},
        "local": {
            "training": {
                "epochs": 3,
            }
        },
    }
    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model="doc2vec"):
    SUPPORTED_DATASETS = ["wikipedia"]
    if dataset in SUPPORTED_DATASETS:
        if model not in "doc2vec":
            dataset = dataset + "_" + model

        data = datahandlers.get_datahandler_config(dataset, folder_data, party_id, is_agg)
    else:
        raise Exception("The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model="doc2vec"):
    SUPPORTED_MODELS = ["doc2vec"]

    if model not in SUPPORTED_MODELS:
        raise Exception("Invalid model config for this fusion algorithm")

    current_module = globals().get("__package__")

    model_module = import_module("{}.model_{}".format(current_module, model))
    method = getattr(model_module, "get_model_config")

    return method(folder_configs, dataset, is_agg=is_agg, party_id=0)
