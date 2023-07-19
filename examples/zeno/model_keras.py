import os

import keras
from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential


def get_hyperparams():
    local_params = {"training": {"epochs": 3}}

    return local_params


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):
    num_classes = 10
    img_rows, img_cols = 28, 28
    if K.image_data_format() == "channels_first":
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=["accuracy"]
    )

    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    # Save model
    fname = os.path.join(folder_configs, "compiled_keras.h5")
    model.save(fname)

    K.clear_session()
    # Generate model spec:
    spec = {"model_name": "keras-cnn", "model_definition": fname}

    model = {"name": "KerasFLModel", "path": "ibmfl.model.keras_fl_model", "spec": spec}

    return model
