# Setup

This tutorial explains how to setup and run IBM federated learning from scratch. All commands are assumed to be run from the base directory at the top of this repository.

## Setup IBM federated learning

To run projects in IBM federated learning, you must first install all the requirements. We highly recommend using Conda installation for this project. If you don't have Conda, you can [install it here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

The latest IBM FL library supports model training using Keras (with TensorFlow v1), TensorFlow v2, PyTorch, and Scikit-learn. It is recommended to install IBM FL in different conda environments for the Keras and TensorFlow v2 versions. Models using PyTorch or Scikit-learn will work on either.

### Installation with Conda (recommended)

1. If you already have Conda installed, create a new environment for IBM FL. We recommend using Python 3.6, but newer versions may also work.

    a. If running experiments using Keras models (with Tensorflow v1), create a new environment by running:

    ```bash
    conda create -n <env_name> python=3.6 tensorflow=1.15
    ```

    b. If running experiments using TensorFlow v2, create a new environment by running:

    ```bash
    conda create -n <env_name> python=3.6
    ```

    c. If running experiments using PyTorch or Scikit-learn, either environment will work.

2. Activate the new Conda environment by running:

    ```bash
    conda activate <env_name>
    ```

    If using TensorFlow v2, install the package:

    ```bash
    pip install tensorflow==2.1.0
    ```

    If this version of TensorFlow is unavailable, try installing a newer version.

3. Install the IBM FL package by running:

    ```bash
    pip install <IBM_federated_learning_whl_file>
    ```

### Installation with virtualenv

We recommend using Python 3.6, but newer versions may also work.

1. Create a virtual environment by running:

    ```bash
    python -m pip install --user virtualenv
    virtualenv venv
    source venv/bin/activate
    python -m pip install --upgrade pip
    ```

2. Install basic dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    If using TensorFlow v2, install the package:

    ```bash
    pip install tensorflow==2.1.0
    ```

    If this version of TensorFlow is unavailable, try installing a newer version.

3. Install the IBM FL package by running:

    ```bash
    pip install <IBM_federated_learning_whl_file>
    ```

## Split Sample Data

You can use `generate_data.py` to generate sample data on any of the integrated datasets. This script requires the following flags:

| Flag | Description | Type |
| - | - | - |
| `-n <num_parties>` | the number of parties to split the data into | integer |
| `-d <dataset>` | which data set to use | string |
| `-pp <points_per_party>` | the number of data points per party | integer |

For example to generate data for **2 parties** with **200 data points** each from the **MNIST dataset**, you could run:

```bash
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
| `-m <model>` | which framework model to use (`keras`, `tf`, `pytorch`, `sklearn`) | string |
| `-n <num_parties>` | the number of parties to split the data into | integer |
| `-d <dataset>` | which data set to use | string |
| `-p <path>` | path to load saved config data | string |

The `-n <num_parties>` and `-d <dataset>` flags should be the same same as when generating the sample data. The `-p <path>` flag will depend on the generated data from the previous step, but will typically be `-p examples/data/<dataset>/random`.

This script will generate config files as follows:

```bash
# aggregator config
examples/configs/<fusion>/<model>/config_agg.yml
# party configs
examples/configs/<fusion>/<model>/config_party0.yml
examples/configs/<fusion>/<model>/config_party1.yml
...
examples/configs/<fusion>/<model>/config_party<n-1>.yml
```

For example to generate the configs for a **Keras model** for **2 parties** using the **iterated average fusion algorithm** from the **MNIST dataset** (generated from before), you could run:

```bash
python examples/generate_configs.py -f iter_avg -m keras -n 2 -d mnist -p examples/data/mnist/random
```

This command will generate the following config files:

```bash
# aggregator config
examples/configs/iter_avg/keras/config_agg.yml
# party configs
examples/configs/iter_avg/keras/config_party0.yml
examples/configs/iter_avg/keras/config_party1.yml
```

Run `python examples/generate_configs.py -h` for full descriptions of the different options.

### Using IBM Cloud interoperability (PubSub Plugin)

A more sophisticated communications mechanism between parties and the aggregator is also available. This is called the PubSub plugin, which is based on the publish/subscribe design pattern. It uses a service broker, which is an IBM Cloud hosted instance of RabbitMQ, backed by a number of cloud micro-services. The purpose of this is to provide a more secure and privacy aware mechanism for running a federated learning task, whereby no party or the aggregator is required to present a service or listen on an open port.

As the service broker is running on IBM Cloud, a user account for the broker is required for the aggregator and each party. You can create accounts as follows:

```bash
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

Note: the user account that creates the federated learning task should be the aggregator.

If the task creation fails, it may be due to a firewall blocking access. In this case, adding a firewall rule allowing access to the `broker_host` field in the aggregator.json file should resolve this.

Now that the correct number of broker user accounts are created and we have a task created, we can generate the configs to use the PubSub plugin:

```bash
python examples/generate_configs.py -f iter_avg -m keras -n 2 -d mnist -p examples/data/mnist/random -c pubsub -t <TASK NAME>
```

Note: The config generation for the PubSub plugin assumes the credentials json file names above, i.e. `aggregator.json`, `party0.json`, `party1.json`, etc.

## Initiate Learning

### Start the Aggregator

To start the aggregator, open a terminal window running the IBM FL environment set up previously.

1. In the terminal run:

    ```bash
    python -m ibmfl.aggregator.aggregator examples/configs/<fusion>/<model>/config_agg.yml
    ```

    where the path provided is the aggregator config file. So using the examples above for the **Keras model** using the **iterated average fusion algorithm**, you can run:

    ```bash
    python -m ibmfl.aggregator.aggregator examples/configs/iter_avg/keras/config_agg.yml
    ```

2. Then in the terminal, type `START` and press enter.

#### Register Parties

To register new parties, open a new terminal window for each party. Activate the conda environment to ensure running the IBM FL environment set up previously.

1. In the terminal run:

    ```bash
    python -m ibmfl.party.party examples/configs/<fusion>/<model>/config_party<idx>.yml
    ```

    where the path provided is the path to the party config file. Each party will have a different config file, usually noted by changing `config_party<idx>.yml`

    So using the examples above for the **Keras model** with **2 parties**, you can run in one terminal:

    ```bash
    python -m ibmfl.party.party examples/configs/iter_avg/keras/config_party0.yml
    ```

    and run in another terminal:

    ```bash
    python -m ibmfl.party.party examples/configs/iter_avg/keras/config_party1.yml
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
