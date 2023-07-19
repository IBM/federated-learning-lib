import os
from importlib import import_module

import keras
from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {"name": "IterAvgFusionHandler", "path": "ibmfl.aggregator.fusion.iter_avg_fusion_handler"}
    return fusion


def get_local_training_config():
    local_training_handler = {
        "name": "FedPlusLocalTrainingHandler",
        "path": "ibmfl.party.training.fedplus_local_training_handler",
    }
    local_training_handler["info"] = {"alpha": 0.1}
    return local_training_handler


def get_hyperparams(model="keras"):
    hyperparams = {
        "global": {"rounds": 2000, "termination_accuracy": 0.83, "max_timeout": 600},
        "local": {"training": {"epochs": 10, "batch_size": 10}, "optimizer": {"lr": 0.0003}},
    }

    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model="keras"):
    SUPPORTED_DATASETS = ["femnist"]
    if dataset in SUPPORTED_DATASETS:
        data = datahandlers.get_datahandler_config(dataset, folder_data, party_id, is_agg)
    else:
        raise Exception("The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model="keras"):
    if is_agg:
        return None

    if model is None or model is "default":
        model = "keras"

    num_classes = 62
    img_rows, img_cols = 28, 28
    if K.image_data_format() == "channels_first":
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation="relu", padding="same", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(
        Conv2D(
            64,
            (5, 5),
            activation="relu",
            padding="same",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(2048, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=["accuracy"])
    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    # Save model
    fname = os.path.join(folder_configs, "compiled_femnist_keras.h5")
    model.save(fname)

    K.clear_session()
    # Generate model spec:
    spec = {"model_name": "keras-cnn", "model_definition": fname}

    model = {"name": "KerasFLModel", "path": "ibmfl.model.keras_fl_model", "spec": spec}

    return model
