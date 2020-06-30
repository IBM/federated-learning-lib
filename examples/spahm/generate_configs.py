import os
import pickle

from sklearn.cluster import KMeans
import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {
        'name': 'SPAHMFusionHandler',
        'path': 'ibmfl.aggregator.fusion.spahm_fusion_handler'
    }
    return fusion


def get_local_training_config():
    local_training_handler = {
        'name': 'LocalTrainingHandler',
        'path': 'ibmfl.party.training.local_training_handler'
    }
    return local_training_handler


def get_hyperparams():
    hyperparams = {
        'global': {
            'rounds': 1,
            'iters': 50,
            'optimize_hyperparams': True
        },
        'local': {
            'training': {
                'max_iter': 500,
                'n_clusters': 10
            }
        }
    }

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False):

    SUPPORTED_DATASETS = ['federated-clustering']
    if dataset in SUPPORTED_DATASETS:
        data = datahandlers.get_datahandler_config(
            dataset, folder_data, party_id, is_agg)
    else:
        raise Exception(
            "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):

    model = KMeans()

    # Save model
    fname = os.path.join(folder_configs, 'kmeans-central-model.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(model, f)
    # Generate model spec:
    spec = {
        'model_name': 'sklearn-kmeans',
        'model_definition': fname
    }

    model = {
        'name': 'SklearnKMeansFLModel',
        'path': 'ibmfl.model.sklearn_kmeans_fl_model',
        'spec': spec
    }

    return model