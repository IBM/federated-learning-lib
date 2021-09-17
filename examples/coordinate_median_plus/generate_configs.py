import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {
        'name': 'CoordinateMedianFedplusFusionHandler',
        'path': 'ibmfl.aggregator.fusion.coordinate_median_fedplus_fusion_handler'
    }
    return fusion


def get_local_training_config(configs_folder=None):
    local_training_handler = {
        'name': 'CoordinateMedianFedPlusLocalTrainingHandler',
        'path': 'ibmfl.party.training.coordinate_median_fedplus_local_training_handler'
    }
    local_training_handler['info'] = {
        'alpha': 0.01,
        'rho': 10
    }
    return local_training_handler


def get_hyperparams(model='keras'):
    hyperparams = {
        'global': {
            'rounds': 3,
            'termination_accuracy': 0.83,
            'max_timeout': 600,
            'rho': 10
        },
        'local': {
            'training': {
                'epochs': 10,
                'batch_size': 10
            },
            'optimizer': {
                'lr': 0.0003
            }
        }
    }

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model='tf'):
    SUPPORTED_DATASETS = ['mnist', 'custom_dataset']
    if dataset in SUPPORTED_DATASETS:
        dataset = dataset + "_" + model

        data = datahandlers.get_datahandler_config(
            dataset, folder_data, party_id, is_agg)
    else:
        raise Exception(
            "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model='tf'):
    if is_agg:
        return None

    if model is None or model is 'default':
        model = 'tf'

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(32, 3, activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(10)

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    # Create an instance of the model
    model = MyModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    acc = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    model.compile(optimizer=optimizer, loss=loss_object, metrics=[acc])
    img_rows, img_cols = 28, 28
    input_shape = (None, img_rows, img_cols, 1)
    model.compute_output_shape(input_shape=input_shape)

    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    model.save(folder_configs)

    spec = {'model_name': 'tf-cnn',
            'model_definition': folder_configs}

    model = {
        'name': 'TensorFlowFLModel',
        'path': 'ibmfl.model.tensorflow_fl_model',
        'spec': spec
    }

    return model
