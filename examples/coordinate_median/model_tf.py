import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten


def get_hyperparams():
    local_params = {"training": {"epochs": 3}}

    return local_params


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):
    if is_agg:
        return None

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(32, 3, activation="relu")
            self.flatten = Flatten()
            self.d1 = Dense(128, activation="relu")
            self.d2 = Dense(10)

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    # Create an instance of the model
    model = MyModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    acc = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    model.compile(optimizer=optimizer, loss=loss_object, metrics=[acc])

    img_rows, img_cols = 28, 28
    input_shape = (None, img_rows, img_cols, 1)
    model.compute_output_shape(input_shape=input_shape)

    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    model.save(folder_configs)

    spec = {"model_name": "tf-cnn", "model_definition": folder_configs}

    model = {"name": "TensorFlowFLModel", "path": "ibmfl.model.tensorflow_fl_model", "spec": spec}

    return model
