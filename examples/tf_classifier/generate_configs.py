import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np

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
        if dataset == 'mnist':
            dataset = 'mnist_tf'
        data = datahandlers.get_datahandler_config(
            dataset, folder_data, party_id, is_agg)
    else:
        raise Exception(
            "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):
    if is_agg:
        return None

    img_rows, img_cols = 28, 28
    batch_size = 28
    input_shape = (batch_size, img_rows, img_cols, 1)
    sample_input = np.zeros(shape=input_shape)

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
    model._set_inputs(sample_input)

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
