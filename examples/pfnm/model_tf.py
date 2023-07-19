import copy
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Reshape


def get_hyperparams():
    local_params = {"training": {"epochs": 3}}

    return local_params


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):
    import copy

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Reshape

    num_classes = 10
    img_rows, img_cols = 28, 28

    input_shape = (img_rows, img_cols, 1)

    model = tf.keras.Sequential()
    model.add(Reshape((784,), input_shape=input_shape))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss_object, optimizer=keras.optimizers.Adadelta(), metrics=["accuracy"])

    # Save model
    fname = os.path.join(folder_configs, "compiled_keras_global.h5")

    # set save_format = 'h5'
    model.save(fname, save_format="h5")

    spec = {"model_name": "tf-cnn", "model_definition": fname}

    model = {"name": "TensorFlowFLModel", "path": "ibmfl.model.tensorflow_fl_model", "spec": spec}

    return model
