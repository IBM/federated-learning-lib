import importlib
import os
from importlib import import_module

if importlib.util.find_spec("pyhelayers") is not None:
    import pyhelayers as pyhe
    from ibmfl.crypto.generate_store_HE_keys import generate_store_HE_keys

import examples.datahandlers as datahandlers

crypto_keys_dir = "examples/keys"


def create_crypto_keys(config_path):
    global crypto_keys_dir

    if config_path is not None:
        crypto_keys_dir = os.path.join(config_path, "keys")

    if not os.path.exists(crypto_keys_dir):
        os.makedirs(crypto_keys_dir)

    # Path for public key context file
    ctx_file = os.path.join(crypto_keys_dir, "fhe.context")
    # Path for secret key context file
    key_file = os.path.join(crypto_keys_dir, "fhe.key")
    # Generate crypto keys
    generate_store_HE_keys(path_for_public_key=ctx_file, path_for_secret_key=key_file)


def get_fusion_config():
    fusion = {
        "name": "CryptoIterAvgFusionHandler",
        "path": "ibmfl.aggregator.fusion.crypto_iter_avg_fusion_handler",
        "info": {"crypto": get_crypto_config(isParty=False)},
    }
    return fusion


def get_local_training_config(configs_folder):
    local_training_handler = {
        "name": "CryptoLocalTrainingHandler",
        "path": "ibmfl.party.training.crypto_local_training_handler",
        "info": {"crypto": get_crypto_config(isParty=True)},
    }
    return local_training_handler


def get_hyperparams(model):
    hyperparams = {
        "global": {
            "rounds": 3,
        }
    }
    current_module = globals().get("__package__")
    model_module = import_module("{}.model_{}".format(current_module, model))
    local_params_method = getattr(model_module, "get_hyperparams")
    local_params = local_params_method()
    hyperparams["local"] = local_params

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model="keras"):
    SUPPORTED_DATASETS = ["mnist", "adult", "cifar10", "femnist", "custom_dataset"]
    if dataset in SUPPORTED_DATASETS:
        if model not in "keras":
            dataset = dataset + "_" + model
        data = datahandlers.get_datahandler_config(dataset, folder_data, party_id, is_agg)
    else:
        raise Exception("The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model="keras"):
    SUPPORTED_MODELS = ["keras", "tf", "pytorch", "sklearn"]

    if model not in SUPPORTED_MODELS:
        raise Exception("Invalid model config for this fusion algorithm")

    current_module = globals().get("__package__")

    model_module = import_module("{}.model_{}".format(current_module, model))
    method = getattr(model_module, "get_model_config")

    return method(folder_configs, dataset, is_agg=is_agg, party_id=0)


def get_crypto_config(isParty):
    global crypto_keys_dir

    # Path for public key context file
    ctx_file = os.path.join(crypto_keys_dir, "fhe.context")
    # Path for secret key context file
    key_file = os.path.join(crypto_keys_dir, "fhe.key")

    if not isParty:
        crypto = {
            "name": "CryptoFHE",
            "path": "ibmfl.crypto.helayer.fhe",
            "key_manager": {
                "name": "LocalDiskKeyManager",
                "path": "ibmfl.crypto.keys_mng.crypto_key_mng_dsk",
                "key_mgr_info": {"files": {"context": ctx_file}},
            },
        }
    else:
        crypto = {
            "name": "CryptoFHE",
            "path": "ibmfl.crypto.helayer.fhe",
            "key_manager": {
                "name": "LocalDiskKeyManager",
                "path": "ibmfl.crypto.keys_mng.crypto_key_mng_dsk",
                "key_mgr_info": {"files": {"context": ctx_file, "key": key_file}},
            },
        }

    return crypto
