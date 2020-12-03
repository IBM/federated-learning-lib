import os


def get_datahandler_config(dh_name, folder_data, party_id, is_agg):
    if dh_name == 'mnist':
        data = {
            'name': 'MnistKerasDataHandler',
            'path': 'ibmfl.util.data_handlers.mnist_keras_data_handler',
            'info': {
                'npz_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.npz')
            }
        }
        if is_agg:
            data['info'] = {
                'npz_file': os.path.join("examples", "datasets", "mnist.npz")
            }
    elif dh_name == 'mnist_dp':
        data = {
            'name': 'MnistDPKerasDataHandler',
            'path': 'ibmfl.util.data_handlers.mnist_keras_data_handler',
            'info': {
                'npz_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.npz')
            }
        }
        if is_agg:
            data['info'] = {
                'npz_file': os.path.join("examples", "datasets", "mnist.npz")
            }

    elif dh_name == 'mnist_sklearn':
        data = {
            'name': 'MnistSklearnDataHandler',
            'path': 'ibmfl.util.data_handlers.mnist_sklearn_data_handler',
            'info': {
                'npz_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.npz')
            }
        }
        if is_agg:
            data['info'] = {
                'npz_file': os.path.join("examples", "datasets", "mnist.npz")
            }

    elif dh_name == 'mnist_unsupervised':
        data = {
            'name': 'MnistSklearnUnsupervisedDataHandler',
            'path': 'ibmfl.util.data_handlers.mnist_sklearn_data_handler',
            'info': {
                'npz_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.npz')
            }
        }
        if is_agg:
            data['info'] = {
                'npz_file': os.path.join("examples", "datasets", "mnist.npz")
            }
    elif dh_name == 'mnist_pytorch':
        data = {
            'name': 'MnistPytorchDataHandler',
            'path': 'ibmfl.util.data_handlers.mnist_pytorch_data_handler',
            'info': {
                'npz_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.npz')
            }
        }
        if is_agg:
            data['info'] = {
                'npz_file': os.path.join("examples", "datasets", "mnist.npz")
            }
    elif dh_name == 'mnist_tf':
        data = {
            'name': 'MnistTFDataHandler',
            'path': 'ibmfl.util.data_handlers.mnist_keras_data_handler',
            'info': {
                'npz_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.npz')
            }
        }
        if is_agg:
            data['info'] = {
                'npz_file': os.path.join("examples", "datasets", "mnist.npz")
            }

    elif dh_name == 'adult':
        data = {
            'name': 'AdultDTDataHandler',
            'path': 'ibmfl.util.data_handlers.adult_dt_data_handler',
            'info': {
                'txt_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.csv')
            }
        }
        if is_agg:
            data['info'] = {
                'txt_file': os.path.join("examples", "datasets", "adult.data")
            }

    elif dh_name == 'adult_sklearn':
        data = {
            'name': 'AdultSklearnDataHandler',
            'path': 'ibmfl.util.data_handlers.adult_sklearn_data_handler',
            'info': {
                'txt_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.csv')
            }
        }
        if is_agg:
            data['info'] = {
                'txt_file': os.path.join("examples", "datasets", "adult.data")
            }

    elif dh_name == 'nursery':
        data = {
            'name': 'NurseryDataHandler',
            'path': 'ibmfl.util.data_handlers.nursery_data_handler',
            'info': {
                'txt_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.csv')
            }
        }
        if is_agg:
            data['info'] = {
                'txt_file': os.path.join("examples", "datasets", "nursery.data")
            }

    elif dh_name == 'higgs':
        data = {
            'name': 'HiggsDataHandler',
            'path': 'ibmfl.util.data_handlers.higgs_data_handler',
            'info': {
                'txt_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.csv')
            }
        }
        if is_agg:
            data['info'] = {}

    elif dh_name == 'airline':
        data = {
            'name': 'AirlineDataHandler',
            'path': 'ibmfl.util.data_handlers.airline_data_handler',
            'info': {'txt_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.csv')}
        }
        if is_agg:
            data['info'] = {}

    elif dh_name == 'diabetes':
        data = {
            'name': 'DiabetesDataHandler',
            'path': 'ibmfl.util.data_handlers.diabetes_data_handler',
            'info': {'txt_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.csv')}
        }
        if is_agg:
            data['info'] = {}

    elif dh_name == 'binovf':
        data = {
            'name': 'BinovfDataHandler',
            'path': 'ibmfl.util.data_handlers.binovf_data_handler',
            'info': {'txt_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.csv')}
        }
        if is_agg:
            data['info'] = {}

    elif dh_name == 'multovf':
        data = {
            'name': 'MultovfDataHandler',
            'path': 'ibmfl.util.data_handlers.multovf_data_handler',
            'info': {'txt_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.csv')}
        }
        if is_agg:
            data['info'] = {}

    elif dh_name == 'multovf_keras':
        data = {
            'name': 'MultovfKerasDataHandler',
            'path': 'ibmfl.util.data_handlers.multovf_data_handler',
            'info': {'txt_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.csv')}
        }

    elif dh_name == 'linovf':
        data = {
            'name': 'LinovfDataHandler',
            'path': 'ibmfl.util.data_handlers.linovf_data_handler',
            'info': {'txt_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.csv')}
        }
        if is_agg:
            data['info'] = {}

    elif dh_name == 'cartpole':
        data = {
            'info': {
                'env_spec': {
                    'env_name': 'CartPoleEnv',
                    'env_definition': 'ibmfl.util.data_handlers.cartpole_env'
                }
            },
            'name': 'CartpoleEnvDataHandler',
            'path': 'ibmfl.util.data_handlers.cartpole_env_data_handler'
        }

    elif dh_name == 'pendulum':
        data = {
            'info': {
                'env_spec': {
                    'env_name': 'PendulumEnv',
                    'env_definition': 'ibmfl.util.data_handlers.pendulum_env'
                }
            },
            'name': 'PendulumEnvDataHandler',
            'path': 'ibmfl.util.data_handlers.pendulum_env_data_handler'
        }

    elif dh_name == 'federated-clustering':
        data = {
            'name': 'FederatedClusteringDataHandler',
            'path': 'ibmfl.util.data_handlers.federated_clustering_data_handler',
            'info': {
                'npz_file': os.path.join(folder_data,
                                         'data_party' + str(
                                             party_id) + '.npz')
            }
        }
        if is_agg:
            data['info'] = {}

    elif dh_name == 'femnist':
        data = {
            'name': 'FemnistKerasDataHandler',
            'path': 'ibmfl.util.data_handlers.femnist_keras_data_handler',
            'info': {
                'npz_file': os.path.join(folder_data, 'data_party' + str(party_id) + '.npz')
            }
        }
        if is_agg:
            data['info'] = {
                'data_folder': os.path.join("examples", "datasets", "femnist")
            }

    return data
