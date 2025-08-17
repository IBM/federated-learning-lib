import os
import joblib
from enum import Enum
from sklearn.linear_model import SGDClassifier

import examples.datahandlers as datahandlers


class Dataset(Enum):
    ADULT = "adult"
    COMPAS = "compas"
    GERMAN = "german"
    CUSTOM_DATASET = "custom_dataset"


class ModelType(Enum):
    SKLEARN = "sklearn"
    # Add other model types here if needed


def get_fusion_config():
    fusion = {"name": "IterAvgFusionHandler", "path": "ibmfl.aggregator.fusion.iter_avg_fusion_handler"}
    return fusion


def get_local_training_config(configs_folder=None):
    local_training_handler = {
        "name": "ReweighLocalTrainingHandler",
        "path": "ibmfl.party.training.reweigh_local_training_handler",
    }
    return local_training_handler


def get_hyperparams(model_type: ModelType):
    hyperparams = {
        "global": {"rounds": 3, "termination_accuracy": 0.9},
        "local": {"training": {"max_iter": 2}}
    }
    return hyperparams


def get_data_handler_config(party_id, dataset: Dataset, folder_data, is_agg=False):
    SUPPORTED_DATASETS = {
        Dataset.ADULT: "adult_sklearn",
        Dataset.COMPAS: "compas_sklearn",
        Dataset.GERMAN: "german_sklearn",
    }

    if dataset not in SUPPORTED_DATASETS:
        raise Exception("The dataset {} is not supported.".format(dataset.value))

    data = datahandlers.get_datahandler_config(SUPPORTED_DATASETS[dataset], folder_data, party_id, is_agg)
    return data


def save_model(model, folder_configs):
    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    fname = os.path.join(folder_configs, "model_architecture.pickle")
    with open(fname, "wb") as f:
        joblib.dump(model, f)

    return fname


def get_model_config(folder_configs, dataset: Dataset, is_agg=False, party_id=0, model_type: ModelType = ModelType.SKLEARN):
    if is_agg:
        return None

    if model_type == ModelType.SKLEARN:
        model = SGDClassifier(loss="log", penalty="l2")
    else:
        raise ValueError("Model type {} is not supported.".format(model_type.value))

    model_spec = {"model_definition": save_model(model, folder_configs)}

    model_config = {
        "name": "SklearnSGDFLModel",
        "path": "ibmfl.model.sklearn_SGD_linear_fl_model",
        "spec": model_spec,
    }

    return model_config
