import os

import tensorflow as tf


def get_hyperparams():
    local_hyperparams = {"training": {"epochs": 2}}
    return local_hyperparams


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):
    if is_agg:
        return None

    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    model = define_mnist_model_via_tf_keras()
    model.save(folder_configs)
    spec = {"model_name": "tf-cnn", "model_definition": folder_configs}
    return {"name": "TensorFlowFLModel", "path": "ibmfl.model.tensorflow_fl_model", "spec": spec}


def define_mnist_model_via_tf_keras():
    num_classes = 10
    img_rows, img_cols = 28, 28

    class MnistCnnModel(tf.keras.Model):
        def __init__(self):
            super(MnistCnnModel, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu")
            self.flatten = tf.keras.layers.Flatten()
            self.d1 = tf.keras.layers.Dense(128, activation="relu")
            self.d2 = tf.keras.layers.Dense(num_classes)

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    model = MnistCnnModel()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    input_shape = (None, img_rows, img_cols, 1)
    model.compute_output_shape(input_shape=input_shape)

    return model
