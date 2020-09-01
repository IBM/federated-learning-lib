# Load large datasets in a FL job

In a FL job, it is possible that the parties' local dataset is large in volume, e.g., image datasets like [Cifar10](https://en.wikipedia.org/wiki/CIFAR-10), 
so that it is impossible to load them all at once into the memory as `np.ndarray` via a data handler.
IBM federated learning supports data generators as an alternative way to load the large datasets 
while performing local training at the party side. 
In this tutorial we will discuss details about 
how to incorporate data generators in your data handler. 


## Set up a data generator for training Keras neural network
As we have discussed in the [Create a Customized Data Handler ](create_my_data_handler.md) tutorial, 
IBM FL uses the `DataHandler` class to load and pre-process data. 
Therefore, we will set up a data generator in our customized data handler to load large datasets.
Below is an example data handler to load [MNIST](https://en.wikipedia.org/wiki/MNIST_database) 
via a `DataGenerator` class defined with Keras.

```python
import os
import glob
import numpy as np
import keras
from keras import backend as K
from matplotlib import image

from ibmfl.data.data_handler import DataHandler


class MnistKerasDataGenerator(DataHandler):

    def __init__(self, data_config):
        super().__init__()
        
        # Specify the directory of training and testing dataset files.
        self.train_file = data_config['training_set']
        self.test_file = data_config['testing_set']
        
        # Set up the data generators with the `DataGenerator` class (see below for its definition)
        self.train_datagenerator = DataGenerator(self.train_file, None, 64)
        self.test_datagenerator = DataGenerator(self.test_file, None, 64)
        
        # Load the batch size if any
        if 'batch_size' in data_config:
            self.batch_size = data_config['batch_size']
            self.set_batch_size(self.batch_size)

    def get_data(self):
        """
        Return a tuple of data generators for training and testing purposes.
        """
        return self.train_datagenerator, self.test_datagenerator

    def set_batch_size(self, batch_size):
        """
        Set up the batch size for loading the training data samples.
        """
        self.train_datagenerator.set_batch_size(batch_size)


# Create a Keras data generator class as one would do for a centralized ML training job. 
# The following code is just a sample implementation. 
# You can implement your `DataGenerator` as per your needs.
class DataGenerator(keras.utils.Sequence):
    """
    Create a data generator that inherits the properties of `keras.utils.Sequence`.
    """

    def __init__(self, directory, labels, batch_size=32):
        """
        Initialization. 
        This data generator assumes features are provided as in `.png` format, 
        and labels are provided as in `string` format. 
        And features will be converted using `image.imread(fname)` into `numpy.ndarray`.
        """
        self.directory = directory # directory where the dataset/features are located 
        self.labels = labels # directory where the label is located
        self.batch_size = batch_size # batch_size to load the data samples
        self.image_shape = (28, 28) # this is the input image size for MNIST
        if not labels:
            labels = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    labels.append(subdir)
            self.labels = labels

        self.num_classes = len(labels)
        self.class_indices = dict(zip(labels, range(len(labels))))
        self.filenames = []
        self.file_classes = []
        for subdir in labels:
            subpath = os.path.join(directory, subdir)
            list_ids = []

            list_ids = list_ids + glob.glob(os.path.join(subpath, '*.png'))
            class_list = [self.class_indices[subdir]] * len(list_ids)
            self.filenames = self.filenames + list_ids
            self.file_classes = self.file_classes + class_list

        self.on_epoch_end()

    # implementing required methods for a Keras data generator
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filenames))

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]
        list_files_temp = [self.filenames[k] for k in indexes]
        list_classes_temp = [self.file_classes[k] for k in indexes]

        X, Y = self.__data_generation(list_files_temp, list_classes_temp)

        return X, Y

    def __data_generation(self, list_files, list_classes):

        batch_x = np.zeros((len(list_files),) +
                           self.image_shape, dtype=K.floatx())
        # build batch of image data
        for i, fname in enumerate(list_files):
            x = image.imread(fname)
            # b = x[:, :, newaxis]
            batch_x[i] = x

        batch_y = keras.utils.to_categorical(
            list_classes, num_classes=self.num_classes)

        batch_x = batch_x.reshape(batch_x.shape[0], 28, 28, 1)

        return batch_x, batch_y
```
**Note that** the `DataGenerator` class in the above example looks the same as 
the `DataGenerator` class in a centralized ML training job. 
If you already had scripts for setting up a `DataGenerator` to train a neural network locally, 
you can reuse it with IBM FL to set up the FL job. 
All other additional code creates a customized data handler inherited from our base data handler class (`DataHandler`), 
like `MnistKerasDataGenerator` in the example above, 
to specify the local data file directory and create the data generators for your training and testing datasets.

```python
# your import statements
# remember to import your DataGenerator class or define it in this file

# imports from ibmfl lib
from ibmfl.data.data_handler import DataHandler


class MyDataHandler(DataHandler):
    """
    Data handler for your data generator.
    """
    def __init__(self, data_config):
        super().__init__()
        # Specify the directory of data files, like
        # self.train_file = data_config['training_set']
        # self.test_file = data_config['testing_set']
        # extract other additional parameters from `info` if any.
        # Set up the data generators with your definition of `DataGenerator` class.

    def get_data(self):
        """
        Gets training and testing data as data generators.
        
        :return: (self.train_datagenerator, self.test_datagenerator) # most build-in training modules expect data is returned in this format
        :rtype: `tuple` 
        """
        pass
```

## Configure the `data` section in config files
Remember to configure the `data` section in your configuration files. 
Below is an example of the `data` section in the configuration files to load the previous data handler, defined `MnistKerasDataGenerator`.
The name and path of the specified data handler must match its relative path in the working directory.
In the `info` section, one can include any information they want to load in their data handler's `__init__`.
```yaml
data:
  info: # load as `data_config`, one can configure the `info` section at will
    training_set: <directory_to_your_training_dataset>
    testing_set: <directory_to_your_testing_datasetd>
    batch_size: 64
  name: MnistKerasDataGenerator # the data handler class name will be loaded
  path: <the_path_where_the_data_handler_class_is_located> 
```
