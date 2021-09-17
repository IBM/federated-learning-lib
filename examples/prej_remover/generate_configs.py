import examples.datahandlers as datahandlers

def get_fusion_config():
    fusion = {
        'name': 'PrejudiceRemoverFusionHandler',
        'path': 'ibmfl.aggregator.fusion.prej_remover_fusion_handler'
    }
    return fusion


def get_local_training_config(configs_folder=None):
    local_training_handler = {
        'name': 'PRLocalTrainingHandler',
        'path': 'ibmfl.party.training.pr_local_training_handler'
    }
    return local_training_handler


def get_hyperparams(model):
    hyperparams = {
        'global': {
            'rounds': 3,
            'termination_accuracy': 0.9
        },
        'local': {
            'training': {
                'max_iter': 2
            }
        }
    }

    return hyperparams

def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model='None'):

    SUPPORTED_DATASETS = ['adult', 'compas', 'custom_dataset']

    if dataset in SUPPORTED_DATASETS:
        if dataset == 'adult':
            dataset = 'adult_pr'
        elif dataset == 'compas':
            dataset = 'compas_pr'
        data = datahandlers.get_datahandler_config(
            dataset, folder_data, party_id, is_agg)
    else:
        raise Exception(
            "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model='None'):
    if is_agg:
        return None

    # Generate model spec:
    spec = {
        'eta': 1,
        'C': 0.0001
    }

    model = {
        'name': 'PrejRemoverFLModel',
        'path': 'ibmfl.model.prej_remover_fl_model',
        'spec': spec
    }

    return model
