"""
Generate config for aggregator and configs
"""
import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {
        'name': 'RLFusionHandler',
        'path': 'ibmfl.aggregator.fusion.rl_avg_fusion_handler'
    }
    return fusion


def get_local_training_config(configs_folder=None):
    local_training_handler = {
        'name': 'RLLocalTrainingHandler',
        'path': 'ibmfl.party.training.rl_local_training_handler'
    }
    return local_training_handler


def get_hyperparams(model):
    hyperparams = {
        'global': {
            'rounds': 1
        }
    }

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model="default"):

    if is_agg:
        return None

    dataset = 'pendulum'
    data = datahandlers.get_datahandler_config(
        dataset, folder_data, party_id, is_agg)

    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model="default"):
    if is_agg:
        return None

    model0 = {
        'name': 'RLlibFLModel',
        'path': 'ibmfl.model.rllib_fl_model',
        'spec': {
                'policy_definition': 'DDPG',
                'policy_name': 'pendulum-ddpg',
                'params': {
                    'training': {
                        'run_config': {
                            'iterations': 1,
                            'checkpoint_frequency': 1
                        },
                        'model_config': {
                            'actor_lr': 0.0001,
                            'num_gpus': 0,
                            'num_workers': 3
                        }
                    },
                    'evaluation': {
                        'run_config': {
                            'steps': 10000
                        }
                    }
                }
        }
    }

    model1 = {
        'name': 'RLlibFLModel',
        'path': 'ibmfl.model.rllib_fl_model',
        'spec': {
            'policy_definition': 'DDPG',
                'policy_name': 'pendulum-ddpg',
                'params': {
                    'training': {
                        'run_config': {
                            'iterations': 1,
                            'checkpoint_frequency': 1
                        },
                        'model_config': {
                            'actor_lr': 0.0001,
                            'num_gpus': 0,
                            'num_workers': 3
                        }
                    },
                    'evaluation': {
                        'run_config': {
                            'steps': 10000
                        }
                    }
                }
        }
    }
    if party_id % 2 == 0:
        model = model0
    else:
        model = model1
    return model
