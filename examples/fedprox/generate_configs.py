import os
from importlib import import_module

import examples.datahandlers as datahandlers
from examples.fedprox.model import MyModel


def get_fusion_config():
    fusion = {"name": "IterAvgFusionHandler", "path": "ibmfl.aggregator.fusion.iter_avg_fusion_handler"}
    return fusion


def get_local_training_config(configs_folder=None):
    local_training_handler = {"name": "LocalTrainingHandler", "path": "ibmfl.party.training.local_training_handler"}
    return local_training_handler


def get_hyperparams(model="tf"):
    hyperparams = {
        "global": {"rounds": 10, "termination_accuracy": 0.9, "max_timeout": 60},
        "local": {"training": {"epochs": 3}},
    }

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model="tf"):
    SUPPORTED_DATASETS = ["mnist", "custom_dataset"]
    if dataset in SUPPORTED_DATASETS:
        if dataset == "mnist":
            dataset = "mnist_tf"
        data = datahandlers.get_datahandler_config(dataset, folder_data, party_id, is_agg)
    else:
        raise Exception("The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model="tf"):
    if is_agg:
        return None

    if model is None or model is "default":
        model = "tf"
    # Create an instance of the model
    model = MyModel()

    # save model to json
    config = model.to_json()
    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)
    fname = os.path.join(folder_configs, "model_architecture.json")
    outfile = open(fname, "w")
    outfile.write(config)
    outfile.close()

    spec = {
        "model_name": "tf-cnn",
        "model_architecture": fname,
        "custom_objects": [{"key": "MyModel", "value": "MyModel", "path": "examples.fedprox.model"}],
        "compile_model_options": {
            "optimizer": {
                "value": "PerturbedGradientDescent",
                "path": "ibmfl.util.fedprox.optimizer",
                "args": {
                    "learning_rate": 0.1,
                    "mu": 0.01,
                },
            },
            "loss": {
                "value": "SparseCategoricalCrossentropy",
                "path": "tensorflow.keras.losses",
                "args": {"from_logits": "true"},
            },
            "metrics": "acc",
        },
    }

    model = {"name": "TensorFlowFLModel", "path": "ibmfl.model.tensorflow_fl_model", "spec": spec}

    return model
