# How to create customized neural network models?

IBM FL allows one to define their neural network to be trained via 
popular machine learning libraries, 
for example, [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/) and 
[PyTorch](https://pytorch.org/).
In this tutorial, we will discuss how to create and plug in a user defined neural network model.

As we have discussed in [Configuring IBM federated learning](configure_fl.md), 
users will specify their models to be trained via the `model` section in the configuration files.
For neural networks, IBM FL supports models defined via one of the following libraries:
* [Keras 2.2.4](#keras-224-and-tensorflow-1150)
* [TensorFlow 1.15.0](#keras-224-and-tensorflow-1150)
* [TensorFlow 2.1.0](#tensorflow-210)
* [PyTorch 1.4.0](#pytorch)

We now go over these supporting libraries one by one.

## Keras 2.2.4 and TensorFlow 1.15.0
To specify a neural network model defined via `Keras (2.2.4)` or `Tensorflow.keras (1.15.0)` 
that will be trained in IBM FL,  
select `name` as `KerasFLModel` and 
`path` as `ibmfl.model.keras_fl_model` in the `model` section of the `.yml` configuration files.
Then, provide the custom model specification in `spec`. 
We now walk you through the process of generating such a model specification.
```python
import os
import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

def generate_model_spec(path_to_save_the_model):
    # Define the model
    num_classes = 10
    img_rows, img_cols = 28, 28
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    # Save the model
    if not os.path.exists(path_to_save_the_model):
        os.makedirs(path_to_save_the_model)

    fname = os.path.join(path_to_save_the_model, 'compiled_keras.h5')
    model.save(fname)

    K.clear_session()
    
    # Define the model specification
    spec = {
        'model_name': 'keras-cnn', # specify your model's name
        'model_definition': fname # specify the path where you saved the model
    }
    return spec
```
In the above example, we define a neural network via Keras and generate 
its corresponding model specification (`spec`). 
There are three main steps to generate `spec`:

1. **Define the model**. Provide the model definition as one usually does via Keras
in a centralized machine learning script.
2. **Save the model**. Save the model as an `h5` file. 
In the above example, the `h5` file is saved at the provided path, `path_to_save_the_model`. 
One can provide their own path to save the `h5` file.
3. **Define the model specification**. 
A model specification is a dictionary with two required keys: `model_name` and `model_definition`, 
where `model_name` is a string specifying a self-defined name, 
and `model_definition` contains the path that stores the `h5` file.

If the model architecture and weights are saved separately, 
you can specify the architecture `JSON` file via key `model_architecture` and 
weights `h5` file via key `model_weights`.
Customized objects can be provided via key `custom_objects`.
You can also specify the compiled options, like `optimizer`, `loss` and `metrics` via key `compiled_options.`

```python
# example for providing model architecture and weights separately
spec = {'model_name': 'keras-cnn',
        'model_architecture': "../model_architecture.json",
        'model_weights': "../model_weights.h5",
        'compile_model_options': 
            {'loss': 'categorical_crossentropy',
             'optimizer': 'adadelta',
             'metrics': ['accuracy']}
        }
            
# example for providing customized objects 
spec = {'model_name': 'keras-cnn',
        'model_definition': "../model.h5",
        'custom_objects': [{'key': 'test_loss', 'value': 'test_loss', 'path': "../loss.py"}]
        }
```
Once the `spec` dictionary is generated, we will provide the `spec` as part of the `model` section 
in the configuration files as shown below:
```yaml
model:
  name: KerasFLModel # for Keras 2.2.4 or TensorFlow.keras 1.15.0
  path: ibmfl.model.keras_fl_model
  spec: spec
```
**Note:** please DO NOT use `Keras` and `tensorflow.keras` at the same time when defining your model. 

## TensorFlow 2.1.0
To specify a neural network model defined via `TensorFlow` or `TensorFlow.keras (2.1.0)` 
that will be trained in IBM FL, in the `model` section of the `.yml` configuration files, select `name` as `TensorFlowFLModel` and `path` as `ibmfl.model.tensorflow_fl_model`. Then, provide the custom model specification in `spec`. IBM FL supports both model formats, i.e., `SavedModel` and `HDF5`, for `TensorFlow 2.1.0`.
We now show examples on how to save the model using each of these formats.

### `SavedModel` format
```python
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np

def generate_model_spec(path_to_save_the_model):
    # Define your model
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
    
    # Save the model
    if not os.path.exists(path_to_save_the_model):
        os.makedirs(path_to_save_the_model)

    model.save(path_to_save_the_model)
    
    # Define the model specification
    spec = {'model_name': 'tf-cnn', # specify your model's name
            'model_definition': path_to_save_the_model # specify the path where you saved the model
            }
    return spec
```
Similar to the previous case, there are three main steps to generate `spec`:

1. **Define the model**. Provide the model definition as one usually does via `TensorFlow 2.1.0`
in a centralized machine learning script. 
In the above example, we take the script from [TensorFlow2 quickstart for experts](https://www.tensorflow.org/tutorials/quickstart/advanced).
2. **Save the model**. Save the model using TensorFlow default format (`SavedModel`). 
In the above example, the model is saved at the provided path, `path_to_save_the_model`. 
One can provide their own path to save the model.
3. **Define the model specification**. 
A model specification is a dictionary with two required keys: `model_name` and `model_definition`, 
where `model_name` is a string specifying a self-defined name, 
and `model_definition` contains the path that stores the model.

### `HDF5` format
In this example, we define a sequential model via `tensorflow.keras (2.1.0)` and 
save it as an `h5` file.
```python
import os
from tensorflow import keras
from tensorflow.keras import layers

def generate_model_spec(path_to_save_the_model):
    # Define the model
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
    # Save the model
    if not os.path.exists(path_to_save_the_model):
        os.makedirs(path_to_save_the_model)

    fname = os.path.join(path_to_save_the_model, 'compiled_tf_keras.h5')
    # set save_format = 'h5'
    model.save(fname, save_format='h5')

    spec = {'model_name': 'tf-cnn',
            'model_definition': fname}
    return spec
```
One can still provide model architecture and weights separately 
if you choose to use `HDF5` format.
Similar to the Keras 2.2.4 case, you will specify the architecture `JSON` file via key `model_architecture` and 
weights `h5` file via key `model_weights`.
Customized objects can be provided via key `custom_objects` for `HDF5` format.
You will specify the compiled options, like `optimizer`, `loss` and `metrics` via key `compiled_options.`


Once the `spec` dictionary is generated, we will provide the `spec` as part of the `model` section 
in the configuration files as shown below:
```yaml
model:
  name: TensorFlowFLModel # for TensorFlow 2.1.0
  path: ibmfl.model.tensorflow_fl_model
  spec: spec
```

## PyTorch 
To specify a neural network model defined via `Pytorch` that will be trained in
IBM FL, int the `model` section of the YAML configuration file,  select `name` 
as `PytorchFLModel` and `path` as `ibmfl.model.pytorch_fl_model`. Then, provide 
the custom model specification in `spec`. We now walk you through the process of 
generating such a model specification for a Pytorch neural network model.

### `SavedModel` format
```python
import os
import torch
from torch import nn


def get_model_config(path_to_save_the_model):

    model = nn.Sequential(nn.Conv2d(1, 32, 3, 1),
                          nn.ReLU(),
                          nn.Conv2d(32, 64, 3, 1),
                          nn.ReLU(),
                          nn.MaxPool2d(2, 2),
                          nn.Dropout2d(p=0.25),
                          nn.Flatten(),
                          nn.Linear(9216, 128),
                          nn.ReLU(),
                          nn.Dropout2d(p=0.5),
                          nn.Linear(128, 10),
                          nn.LogSoftmax(dim=1)
                          )

    if not os.path.exists(path_to_save_the_model):
        os.makedirs(path_to_save_the_model)

    # Save model
    fname = os.path.join(path_to_save_the_model, 'pytorch_sequence.pt')
    torch.save(model, fname)

    # Optional, specify an optimizer class as optim.<optimizer> 
    # The entire expression should be of type string
    # e.g., optimizer = 'optim.SGD'
    optimizer = 'optim.Adam'
    # Optional, specify a loss criterion as nn.<loss-criterion>
    # The entire expression should be of type string
    # e.g., criterion = 'nn.NLLLoss'
    criterion = 'nn.NLLLoss'

    spec = {
        'model_name': 'pytorch-nn',
        'model_definition': fname,
        'optimizer': optimizer,
        'loss_criterion': criterion,
    }
    return spec
```
Similar to the previous case, there are three main steps to generate `spec`:

1. **Define the model**. Provide an nn.Sequential model definition as one usually does via Pytorch
in a centralized machine learning script. 

2. **Save the model**. Save the entire model using torch.save.. 
In the above example, the model is saved at the provided path, `path_to_save_the_model`. 
One can provide their own path to save the model.
3. **Define the model specification**. 
A model specification is a dictionary with two required keys: `model_name` and `model_definition`, 
where `model_name` is a string specifying a self-defined name, 
and `model_definition` contains the path that stores the model.

