import os
# TensorFlow and tf.keras
from tensorflow import keras
from tensorflow.keras import layers


import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {
        'name': 'IterAvgFusionHandler',
        'path': 'ibmfl.aggregator.fusion.iter_avg_fusion_handler'
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
            'rounds': 3,
            'termination_accuracy': 0.9,
            'max_timeout': 60
        },
        'local': {
            'training': {
                'epochs': 3
            }
        }
    }

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False):

    SUPPORTED_DATASETS = ['mnist']
    if dataset in SUPPORTED_DATASETS:
        data = datahandlers.get_datahandler_config(
            dataset, folder_data, party_id, is_agg)
    else:
        raise Exception(
            "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):
    if is_agg:
        return None

    # tf.keras model
    num_classes = 10
    input_shape = (28, 28, 1)
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    fname = os.path.join(folder_configs, 'compiled_tf_keras.h5')
    # set save_format = 'h5'
    model.save(fname, save_format='h5')

    spec = {'model_name': 'tf-cnn',
            'model_definition': fname}

    model = {
        'name': 'TensorFlowFLModel',
        'path': 'ibmfl.model.tensorflow_fl_model',
        'spec': spec
    }

    return model
