# Setup

This tutorial explains how to setup and run IBM federated learning from scratch. All commands are assumed to be run from the base directory at the top of this repository.

## Setup IBM federated learning

To run projects in IBM federated learning, you must first create a Python environment to install all the requirements. You can use either Conda or venv:

<details>
<summary>Conda (recommended)</summary>

If you don't have Conda, you can install it [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install).

Once installed, create a new Conda environment. We recommend using Python 3.7, but newer versions may also work. For Mac M1/M2 systems, you must use Python 3.8 or above.

```sh
conda create -n <env_name> python=3.7
```

Activate the newly created Conda environment.

```sh
conda activate <env_name>
```

</details>

<details>
<summary>Venv</summary>

Create a new virtual environment using Python's built-in `venv` module. This will use your system's Python version which may or may not be fully compatible.

```bash
python -m venv venv
```

Activate the newly created virtual environment.

```sh
source venv/bin/activate
```

</details>

After creating and activating the Python environment, install the wheel file to install the IBM federated learning library and all dependencies. This file is located in the `federated-learning-lib` directory of this repo. By default, the wheel file will not install any additional machine learning libraries. You must specify the desired model training backend library in brackets. The following backends are supported:

```sh
# Install with no additional machine learning libraries
pip install "/path/to/federated_learning_lib.whl"
# Install with Scikit-learn backend
pip install "/path/to/federated_learning_lib.whl[sklearn]"
# Install with PyTorch backend
pip install "/path/to/federated_learning_lib.whl[pytorch]"
# Install with Keras (TensorFlow v1) backend
pip install "/path/to/federated_learning_lib.whl[keras]"
# Install with TensorFlow v2 backend
pip install "/path/to/federated_learning_lib.whl[tf]"
# Install with RLlib backend
pip install "/path/to/federated_learning_lib.whl[rllib]"
```

You can also install multiple backends using a comma separated list. For example:

```sh
# Install with Scikit-learn and Keras (TensorFlow v1) backend
pip install "/path/to/federated_learning_lib.whl[sklearn,keras]"
# Install with PyTorch and TensorFlow v2 backend
pip install "/path/to/federated_learning_lib.whl[pytorch,tf]"
# Install with Scikit-learn, TensorFlow v2, and RLlib backend
pip install "/path/to/federated_learning_lib.whl[sklearn,tf,rllib]"
```

You may install as many backends as you'd like. The only exception is that you **cannot** install both the Keras and TensorFlow v2 backends since they are different versions of TensorFlow.

**Notes:**

* The quotes are required if using the Zsh shell (this is the default shell for Mac).
* There should be no spaces before or after each comma.
* The Keras backend will only work for Python 3.7; therefore, it will not work for Mac M1/M2 systems.

## Split Sample Data

You can use `generate_data.py` to generate sample data on any of the integrated datasets. This script requires the following flags:

| Flag | Description | Type |
| - | - | - |
| `-n <num_parties>` | the number of parties to split the data into | integer |
| `-d <dataset>` | which data set to use | string |
| `-pp <points_per_party>` | the number of data points per party | integer |

For example to generate data for **2 parties** with **200 data points** each from the **MNIST dataset**, you could run:

```sh
python examples/generate_data.py -n 2 -d mnist -pp 200
```

Run `python examples/generate_data.py -h` for full descriptions of the different options.

By default the data is scaled down to range between 0 and 1 and reshaped such that each image is (28, 28). For more information on what preprocessing was performed, check the [Keras classifier example](/examples/keras_classifier).

## Create Configuration Files

To run IBM federated learning, you must have configuration files for the aggregator and for each party.

You can generate these config files using the `generate_configs.py` script. This script requires the following flags:

| Flag | Description | Type |
| - | - | - |
| `-f <fusion>` | which fusion algorithm to run | string |
| `-m <model>` | which framework model to use (`sklearn`, `pytorch`, `keras`, `tf`) | string |
| `-n <num_parties>` | the number of parties to split the data into | integer |
| `-d <dataset>` | which data set to use | string |
| `-p <path>` | path to load saved config data | string |

The `-n <num_parties>` and `-d <dataset>` flags should be the same same as when generating the sample data. The `-p <path>` flag will depend on the generated data from the previous step, but will typically be `-p examples/data/<dataset>/random`. The backend framework for model from the `-m <model>` flag must be installed.

This script will generate config files as follows:

```sh
# aggregator config
examples/configs/<fusion>/<model>/config_agg.yml
# party configs
examples/configs/<fusion>/<model>/config_party0.yml
examples/configs/<fusion>/<model>/config_party1.yml
...
examples/configs/<fusion>/<model>/config_party<n-1>.yml
```

For example to generate the configs for a **PyTorch model** for **2 parties** using the **iterated average fusion algorithm** from the **MNIST dataset** (generated from before), you could run:

```sh
python examples/generate_configs.py -f iter_avg -m pytorch -n 2 -d mnist -p examples/data/mnist/random
```

This command will generate the following config files:

```sh
# aggregator config
examples/configs/iter_avg/pytorch/config_agg.yml
# party configs
examples/configs/iter_avg/pytorch/config_party0.yml
examples/configs/iter_avg/pytorch/config_party1.yml
```

Run `python examples/generate_configs.py -h` for full descriptions of the different options.

### Using IBM Cloud interoperability (PubSub Plugin)

<details>
<summary>Show</summary>

A more sophisticated communications mechanism between parties and the aggregator is also available. This is called the PubSub plugin, which is based on the publish/subscribe design pattern. It uses a service broker, which is an IBM Cloud hosted instance of RabbitMQ, backed by a number of cloud micro-services. The purpose of this is to provide a more secure and privacy aware mechanism for running a federated learning task, whereby no party or the aggregator is required to present a service or listen on an open port.

As the service broker is running on IBM Cloud, a user account for the broker is required for the aggregator and each party. You can create accounts as follows:

```sh
python examples/pubsub_register.py --credentials=pubsub_credentials.json --user=<AGGREGATOR USER> --password=<PASSWORD> > aggregator.json
python examples/pubsub_register.py --credentials=pubsub_credentials.json --user=<PARTY 0> --password=<PASSWORD> > party0.json
python examples/pubsub_register.py --credentials=pubsub_credentials.json --user=<PARTY N> --password=<PASSWORD> > partyn.json
```

In these examples, the output of the registration process is saved to a new json file. In these files there will be specific credentials for each party/aggregator to use during federated learning.

It is also possible to deregister a created account:

```bash
python examples/pubsub_deregister.py --credentials=aggregator.json
```

The PubSub plugin operates on the basis that a Federated Learning task exists. This task can be created as follows:

```bash
python examples/pubsub_task.py --credentials=aggregator.json --task_name=<TASK NAME>
```

**Note:** the user account that creates the federated learning task should be the aggregator.

If the task creation fails, it may be due to a firewall blocking access. In this case, adding a firewall rule allowing access to the `broker_host` field in the aggregator.json file should resolve this.

Now that the correct number of broker user accounts are created and we have a task created, we can generate the configs to use the PubSub plugin:

```sh
python examples/generate_configs.py -f iter_avg -m keras -n 2 -d mnist -p examples/data/mnist/random -c pubsub -t <TASK NAME>
```

**Note:** The config generation for the PubSub plugin assumes the credentials json file names above, i.e. `aggregator.json`, `party0.json`, `party1.json`, etc.
</details>

## Initiate Learning

### Start the Aggregator

To start the aggregator, open a terminal window running the IBM FL environment set up previously.

1. In the terminal run:

    ```sh
    python -m ibmfl.aggregator.aggregator examples/configs/<fusion>/<model>/config_agg.yml
    ```

    where the path provided is the aggregator config file. So using the examples above for the **PyTorch model** using the **iterated average fusion algorithm**, you can run:

    ```sh
    python -m ibmfl.aggregator.aggregator examples/configs/iter_avg/pytorch/config_agg.yml
    ```

2. Then in the terminal, type `START` and press enter.

#### Register Parties

To register new parties, open a new terminal window for each party. Activate the conda environment to ensure running the IBM FL environment set up previously.

1. In the terminal run:

    ```sh
    python -m ibmfl.party.party examples/configs/<fusion>/<model>/config_party<idx>.yml
    ```

    where the path provided is the path to the party config file. Each party will have a different config file, usually noted by changing `config_party<idx>.yml`

    So using the examples above for the **PyTorch model** with **2 parties**, you can run in one terminal:

    ```sh
    python -m ibmfl.party.party examples/configs/iter_avg/pytorch/config_party0.yml
    ```

    and run in another terminal:

    ```sh
    python -m ibmfl.party.party examples/configs/iter_avg/pytorch/config_party1.yml
    ```

2. In the terminal for each party, type `START` and press enter.

3. Then in then terminal for each party, type `REGISTER` and press enter.

### Train and Evaluate the Models

Now that the aggregator and parties are running, we will train and evaluate the model.

1. To initiate federated training, type `TRAIN` in your aggregator terminal and press enter. Each of the parties will now begin training in their respective terminals.

2. Once training is complete, type `EVAL` in each of the party terminals and press enter to evaluate their local model.

### (Optional) Train again, Sync, and Save the Models

You can now enter `TRAIN` again at the aggregator's terminal to continue the FL training.

Alternatively, entering `SYNC` at the aggregator's terminal will trigger the synchronization of the current global model with parties. This can be followed by entering `EVAL` in any of the party terminals to evaluate global model on their local partition of the dataset.

Running `SAVE` at each of the party terminals will trigger the corresponding party to save their model at the local working directory.

### Terminate

Once the training and evaluation is complete, type `STOP` in the aggregator and each party terminals to stop the connections and exit.

## IBM FL Command Reference

| IBM FL Command | Participant | Description |
| :-----------: | :-----------: | :----------- |
| `START` | aggregator / party | Start accepting connections|
| `REGISTER` | party | Join an FL project |
| `TRAIN` | aggregator | Initiate training process |
| `SYNC` | aggregator | Synchronize model among parties |
| `EVAL` | party | Evaluate model |
| `SAVE` | party | Save model in current directory |
| `STOP` | aggregator / party | Stop the connection |
