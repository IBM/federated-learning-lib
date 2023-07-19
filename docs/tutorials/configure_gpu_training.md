# Enabling GPU training

IBM federated learning offers support for training neural network models
under GPU environment at the party side to speedup the training process.

## Environment setup

Please install required libraries for GPU training.

- For Keras and TensorFlow models, install the corresponding `tensorflow-gpu` package
 according to [Tensorflow GPU tutorial](https://www.tensorflow.org/install/gpu).
 IBM FL currently requires `tensorflow==1.15.0`, therefore,
 you will need to install `tensorflow-gpu==1.15.0` in your GPU environment.

## IBM FL configuration

Users can enable and specify the number of GPUs they want to use for training
via the party's configuration file.
Below is an example of the party's configuration file:

```yaml
aggregator:
  ip: 127.0.0.1
  port: 5000
connection:
  info:
    ip: 127.0.0.1
    port: 8085
    tls_config:
      enable: false
  name: FlaskConnection
  path: ibmfl.connection.flask_connection
  sync: false
data:
  info:
    npz_file: examples/data/mnist/random/data_party0.npz
  name: MnistKerasDataHandler
  path: ibmfl.util.data_handlers.mnist_keras_data_handler
local_training:
  name: LocalTrainingHandler
  path: ibmfl.party.training.local_training_handler
model:
  name: KerasFLModel
  path: ibmfl.model.keras_fl_model
  spec:
    model_definition: examples/configs/iter_avg/keras/compiled_keras.h5
    model_name: keras-cnn
  info:
    gpu:
      num_gpus: 2 # enabling keras training with 2 GPUs
protocol_handler:
  name: PartyProtocolHandler
  path: ibmfl.party.party_protocol_handler
```

In the above example, the `gpu` section under `info` section of `model` specifies
the `gpu` setting of party's local training.
Users can change the `num_gpus` according to the computing resources available to the parties.

If no `gpu` section is presented in `info`, the Keras/TensorFlow.keras training will be
using the default CPU environment or **only one GPU** even if the party can access one or more GPU(s).
