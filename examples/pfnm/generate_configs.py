import os

import keras
from keras import backend as K
from keras.layers import Dense, Dropout, Reshape
from keras.models import Sequential

import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {
        'name': 'PFNMFusionHandler',
        'path': 'ibmfl.aggregator.fusion.pfnm_fusion_handler'
    }
    return fusion


def get_local_training_config():
    local_training_handler = {
        'name': 'PFNMLocalTrainingHandler',
        'path': 'ibmfl.party.training.pfnm_local_training_handler'
    }
    return local_training_handler


def get_hyperparams():
    hyperparams = {
        'global': {
            'rounds': 3,
            'termination_accuracy': 0.9
        },
        'local': {
            'training': {
                'epochs': 3
            },
            'optimizer': {
                'lr': 0.01
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
    num_classes = 10
    img_rows, img_cols = 28, 28
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Reshape((784,), input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Save model
    fname = os.path.join(folder_configs, 'compiled_keras_global.h5')
    model.save(fname)

    K.clear_session()

    # Generate model spec:
    spec = {
        'model_name': 'keras-fc',
        'model_definition': fname
    }

    model = {
        'name': 'KerasFLModel',
        'path': 'ibmfl.model.keras_fl_model',
        'spec': spec
    }

    return model
