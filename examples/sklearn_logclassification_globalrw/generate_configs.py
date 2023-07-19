import os

import joblib
from sklearn.linear_model import SGDClassifier

import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {"name": "ReweighFusionHandler", "path": "ibmfl.aggregator.fusion.reweigh_fusion_handler"}
    return fusion


def get_local_training_config(configs_folder=None):
    local_training_handler = {
        "name": "ReweighLocalTrainingHandler",
        "path": "ibmfl.party.training.reweigh_local_training_handler",
    }
    return local_training_handler


def get_hyperparams(model):
    hyperparams = {"global": {"rounds": 3, "termination_accuracy": 0.9}, "local": {"training": {"max_iter": 2}}}

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model="sklearn"):
    SUPPORTED_DATASETS = ["adult", "compas", "german", "custom_dataset"]

    if dataset in SUPPORTED_DATASETS:
        if dataset == "adult":
            dataset = "adult_sklearn_grw"
        elif dataset == "compas":
            dataset = "compas_sklearn_grw"
        elif dataset == "german":
            dataset = "german_sklearn_grw"
        data = datahandlers.get_datahandler_config(dataset, folder_data, party_id, is_agg)
    else:
        raise Exception("The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model="sklearn"):
    if is_agg:
        return None

    model = SGDClassifier(loss="log", penalty="l2")

    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    fname = os.path.join(folder_configs, "model_architecture.pickle")

    with open(fname, "wb") as f:
        joblib.dump(model, f)

    # Generate model spec:
    spec = {"model_definition": fname}

    model = {"name": "SklearnSGDFLModel", "path": "ibmfl.model.sklearn_SGD_linear_fl_model", "spec": spec}

    return model
