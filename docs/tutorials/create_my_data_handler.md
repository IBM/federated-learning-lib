# Create a customized data handler

In IBM federated learning, we use the `DataHandler` class to load and pre-process data. 
When running an FL job, the data from each party must be formatted correctly. 
For this reason, although each party can implement and use different data handlers, 
we recommend parties in the same task use the same or similar data handlers to 
make sure their input data is in the correct format.


As we have discussed in the [previous tutorial](configure_fl.md), 
the aggregator and parties use configuration files (`.yml` files) for initialization 
([Here](configure_fl.md#the-aggregators-configuration-file) is an examples of the configuration files.)
In the config files, a section named `data` will be used to initialize the `DataHandler` class 
for the aggregator (if it has access to any dataset) and parties. 
In particular, both the aggregator and each party can specify their own data handlers in their config files. 
Note that the `data` section is optional in the config file for the aggregator. 
If the aggregator has access to a dataset, for example, a global testset, 
it can access the data via its data handler to monitor the global model's performance. 
The data handlers for each party help them to access their own training and testing data.

## What's inside our build-in data handlers?
Below is one of our build-in data handlers for preparing [MNIST](http://yann.lecun.com/exdb/mnist/) data to train a Keras CNN model, 
see our example [keras_classifier](../../examples/keras_classifier).

The `get_data` method is where the party loads its local dataset to perform training and testing. 
Specifically,  when local training is triggered, the party will load the training data from the first return argument of `get_data`. 
When an evaluation of the model is triggered, the testing data is taken from the second return argument.
If we observe this example code, we find that it loads and pre-processes the MNIST dataset the same way 
we do in a centralized machine learning job. 

**Note that** this data handler assumes the local data is saved as a `.npz` file, and MNIST data has not been pre-processed yet.
However, `.npz` format is not a mandatory format. One can load their local dataset from 
any other file format, like `.csv` and `.txt`, etc. 

```python
import numpy as np

# imports from ibmfl
from ibmfl.data.data_handler import DataHandler
from ibmfl.exceptions import FLException



class MnistKerasDataHandler(DataHandler):
    """
    Data handler for MNIST dataset.
    """

    def __init__(self, data_config=None, channels_first=False):
        super().__init__()
        self.file_name = None
        # `data_config` loads anything inside the `info` part of the `data` section. 
        if data_config is not None:
            # this example assumes the local dataset is in .npz format, so it searches for it.
            if 'npz_file' in data_config: 
                self.file_name = data_config['npz_file']
        self.channels_first = channels_first
        
        if self.file_name is None:
            raise FLException('No data file name is provided to load the dataset.')

    def get_data(self):
        """
        Gets pre-process mnist training and testing data. 

        :return: training and testing data
        :rtype: `tuple`
        """
        num_classes = 10
        img_rows, img_cols = 28, 28
        if self.file_name is None:
            raise FLException('No data file name is provided to load the dataset.')
        else:
            try:
                data_train = np.load(self.file_name)
                x_train = data_train['x_train']
                y_train = data_train['y_train']
                x_test = data_train['x_test']
                y_test = data_train['y_test']
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' +
                              self.file_name)

        if self.channels_first:
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = np.eye(num_classes)[y_train]
        y_test = np.eye(num_classes)[y_test]
        return (x_train, y_train), (x_test, y_test)
```
Below is an example of the `data` section in the configuration files. 
The name and path of the specified data handler must match its relative path in the working directory.
In the `info` section, one can include any information they want to load in their data handler `__init__`.
```yaml
data:
  info: # load as `data_config`, one can configure the `info` section at will
    npz_file: examples/data/mnist/random/data_party0.npz 
  name: MnistKerasDataHandler # the data handler class name will be loaded
  path: datahandlers.mnist_keras_data_handler # the path where the data handler is implemented
```

## How to create my own data handlers for supervised learning?

One can create a customized data handler via inheritance from the ibmfl base data handler class.
In the `__init__` function, it loads the dataset information via `data_config`. 
This argument takes as input a dictionary that is received from the `info` field of the `data` section in the configuration file. 
In the example below, we can see the data file is specified in the config and can be received by the data handler. 
Any other additional hyper-parameters and other arguments can also be passed in here to the data handler from the configuration file.

```python
# your import statements

# imports from ibmfl lib
from ibmfl.data.data_handler import DataHandler

class MyDataHandler(DataHandler):
    """
    Data handler for your dataset.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if '<your_data_file_name>' in data_config:
                self.file_name = data_config['<your_data_file_name>']
            # extract other additional parameters from `info` if any.
    
    def get_data(self):
        """
        Gets training and testing data and performs any pre-processing steps if needed.
        
        :return: ((x_train, y_train), (x_test, y_test)) # most build-in training modules expect data is returned in this format
        :rtype: `tuple` 
        """
```

## IBM FL built-in pre-processing helper functions

IBM FL provides a list of helper functions to facilitate the data preparation process; 
see our [API documentation](http://ibmfl-api-docs.mybluemix.net/index.html) to learn more about `ibmfl.data`.
Users can get it for free by inheriting from `ibmfl.data.data_handler.DataHandler` class. 
All of the current helper functions assume that the input data is of type `numpy.ndarray`.