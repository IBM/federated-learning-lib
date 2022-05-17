This README explains how to setup and run IBM federated learning from scratch. All commands are assumed to be
run from the base directory at the top of this repository.

## Setup IBM federated learning

To run projects in IBM federated learning, you must first install all the requirements. 
We highly recommend using Conda installation for this project. If you don't have Conda,
you can [install it here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

#### Installation with Conda (recommended)

1. If you already have Conda installed, create a new environment for IBM FL by running:

    `conda create -n <env_name> python=3.6`

    **Note**: Latest IBM FL library supports Keras model training with two different 
    Tensorflow Backend versions(1.15 and 2.1). It is recommended to install IBM FL 
    in different conda environment with different tf versions.
    
    a. While running Keras experiments with Tensorflow v1.15, create a new environment 
    by running:

        `conda create -n <env_name> python=3.6 tensorflow=1.15`

    b. While running Keras experiments with Tensorflow v2.1, try creating a new environment by running:

        `conda create -n <env_name> python=3.6 tensorflow=2.1.0`
 
    **Note**: Tensorflow v2.1 may not be available through conda install. If you get a `PackagesNotFoundError` after running the above command, please try creating a new envirnoment via:
        `conda create -n <env_name> python=3.6`
    After activating the new Conda environment (see Step 2), use `pip install tensorflow==2.1` to install the required tensorflow package.

2. Run `conda activate <env_name>` to activate the new Conda environment.

3. Install the IBM FL package by running:
    
    `pip install <IBM_federated_learning_whl_file>`


#### Installation with pip

1. Create a virtual environment by running:

    ```commandline
    python -m pip install --user virtualenv
    virtualenv venv
    source venv/bin/activate
    python -m pip install --upgrade pip
    ```

    **Then run 'source/venv/bin/activate' to enable the virtual environment.**

2. Install basic dependencies:

    `pip install -r requirements.txt`

3. Install the IBM FL package by running:
    
    `pip install <IBM_federated_learning_whl_file>`


## Split Sample Data

You can use `generate_data.py` to generate sample data on any of the integrated datasets. For example, you could run:
```commandline
python examples/generate_data.py -n 2 -d mnist -pp 200
```

This command would generate 2 parties with 200 data points each from the MNIST dataset. By default
the data is scaled down to range between 0 and 1 and reshaped such that each image is (28, 28). For
more information on what preprocessing was performed, check the [Keras classifier example](/examples/keras_classifier).

Run `python examples/generate_data.py -h` for full descriptions
of the different options. 

## Create Configuration Files

To run IBM federated learning, you must have configuration files for the aggregator and for each party.

You can generate these config files using the `generate_configs.py` script.
 
For example, you could run:

```commandline
python examples/generate_configs.py -f iter_avg -m keras -n 2 -d mnist -p examples/data/mnist/random 
```

This command would generate the configs for the `keras_classifier` model, assuming 2 parties.
You must also specify the party data path via `-p`. 

Run `python examples/generate_configs.py -h` for full descriptions of the different options.

#### Using IBM Cloud interoperability (PubSub Plugin)

A more sophisticated communications mechanism between parties and the aggregator is also available. This is called the PubSub plugin, which is based on the publish/subscribe design pattern. It uses a service broker, which is an IBM Cloud hosted instance of RabbitMQ, backed by a number of cloud micro-services. The purpose of this is to provide a more secure and privacy aware mechanism for running a federated learning task, whereby no party or the aggregator is required to present a service or listen on an open port.

As the service broker is running on IBM Cloud, a user account for the broker is required for the aggregator and each party. You can create accounts as follows:

```
python examples/pubsub_register.py --credentials=pubsub_credentials.json --user=<AGGREGATOR USER> --password=<PASSWORD> > aggregator.json
python examples/pubsub_register.py --credentials=pubsub_credentials.json --user=<PARTY 0> --password=<PASSWORD> > party0.json
python examples/pubsub_register.py --credentials=pubsub_credentials.json --user=<PARTY N> --password=<PASSWORD> > partyn.json
```

In these examples, the output of the registration process is saved to a new json file. In these files there will be specific credentials for each party/aggregator to use during federated learning.

It is also possible to deregister a created account:

```
python examples/pubsub_deregister.py --credentials=aggregator.json
```

The PubSub plugin operates on the basis that a Federated Learning task exists. This task can be created as follows:

```
python examples/pubsub_task.py --credentials=aggregator.json --task_name=<TASK NAME>
```

Note: the user account that creates the federated learning task should be the aggregator.

If the task creation fails, it may be due to a firewall blocking access. In this case, adding a firewall rule allowing access to the `broker_host` field in the aggregator.json file should resolve this.

Now that the correct number of broker user accounts are created and we have a task created, we can generate the configs to use the PubSub plugin:

```
python examples/generate_configs.py -f iter_avg -m keras -n 2 -d mnist -p examples/data/mnist/random -c pubsub -t <TASK NAME>
```

Note: The config generation for the PubSub plugin assumes the credentials json file names above, i.e. aggregator.json, party0.json etc.

## Initiate Learning

#### Start the Aggregator

To start the aggregator, open a terminal window running the IBM FL environment set up previously.

1. In the terminal run:
    ```commandline
    python -m ibmfl.aggregator.aggregator examples/configs/iter_avg/keras/config_agg.yml
    ```  

    where the path provided is the aggregator config file path.

2. Then in the terminal, type `START` and press enter.

#### Register Parties

To register new parties, open a new terminal window for each party, running the IBM FL environment set up previously.

1. In the terminal run:
     ```commandline
    python -m ibmfl.party.party examples/configs/iter_avg/keras/config_party0.yml
    ``` 

    where the path provided is the path to the party config file.

    *NOTE*: Each party will have a different config file, usually noted by changing `config_party<idx>.yml`

2.  In the terminal for each party, type `START` and press enter.

3. Then in then terminal for each party, type `REGISTER` and press enter.

#### Train

To initiate federated training, type `TRAIN` in your aggregator terminal and press enter.


## FL Command Reference


| FL Command | Participant | Description |
| :-----------: | :-----------: | :----------- |
| `START` | aggregator / party | Start accepting connections|
| `REGISTER` | party | Join an FL project |
| `TRAIN` | aggregator | Initiate training process |
| `SYNC` | aggregator | Synchronize model among parties |
| `STOP` (coming soon) | aggregator | Pause training process |
| `EVAL` | party | Evaluate model |
