"""
Generate config for aggregator and configs
"""

import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {"name": "RLFusionHandler", "path": "ibmfl.aggregator.fusion.rl_avg_fusion_handler"}
    return fusion


def get_local_training_config(configs_folder=None):
    local_training_handler = {
        "name": "RLLocalTrainingHandler",
        "path": "ibmfl.party.training.rl_local_training_handler",
    }
    return local_training_handler


def get_hyperparams(model="default"):
    hyperparams = {"global": {"rounds": 1}}

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model="default"):
    if is_agg:
        return None

    dataset = "cartpole"
    data = datahandlers.get_datahandler_config(dataset, folder_data, party_id, is_agg)

    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model="default"):
    if is_agg:
        return None

    model0 = {
        "name": "RLlibFLModel",
        "path": "ibmfl.model.rllib_fl_model",
        "spec": {
            "policy_definition": "ppo",
            "policy_name": "cartpole-ppo",
            "params": {
                "training": {
                    "run_config": {"iterations": 1, "checkpoint_frequency": 1},
                    "model_config": {"lr": 0.0005, "num_gpus": 0, "num_workers": 3},
                    "env_config": {
                        "pole_angle_min": 0.2,
                        "pole_angle_max": 0.4,
                        "cart_position_min": 1.5,
                        "cart_position_max": 2.0,
                    },
                },
                "evaluation": {
                    "run_config": {"steps": 10000},
                    "env_config": {
                        "pole_angle_min": -0.2,
                        "pole_angle_max": 0.2,
                        "cart_position_min": -1.0,
                        "cart_position_max": 1.0,
                    },
                },
            },
        },
    }

    model1 = {
        "name": "RLlibFLModel",
        "path": "ibmfl.model.rllib_fl_model",
        "spec": {
            "policy_definition": "ppo",
            "policy_name": "cartpole-ppo",
            "params": {
                "training": {
                    "run_config": {"iterations": 1, "checkpoint_frequency": 1},
                    "model_config": {"lr": 0.0005, "num_gpus": 0, "num_workers": 3},
                    "env_config": {
                        "pole_angle_min": -0.4,
                        "pole_angle_max": -0.2,
                        "cart_position_min": -2.0,
                        "cart_position_max": -1.5,
                    },
                },
                "evaluation": {
                    "run_config": {"steps": 10000},
                    "env_config": {
                        "pole_angle_min": -0.2,
                        "pole_angle_max": 0.2,
                        "cart_position_min": -1.0,
                        "cart_position_max": 1.0,
                    },
                },
            },
        },
    }
    if party_id % 2 == 0:
        model = model0
    else:
        model = model1
    return model
