import copy

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten


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

    def get_config(self):
        layer_configs = []
        for layer in super(MyModel, self).layers:
            layer_configs.append(tf.keras.utils.serialize_keras_object(layer))

        config = {"name": self.name, "layers": copy.deepcopy(layer_configs)}

        return config

    @classmethod
    def from_config(cls, config):
        layer_configs = config["layers"] if "name" in config else config
        model = cls()
        for i, layer in enumerate(model.layers):
            layer.from_config(layer_configs[i]["config"])

        return model
