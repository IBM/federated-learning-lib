This README explains how to setup and run IBM federated learning from scratch. All commands are assumed to be
run from the base directory at the top of this repository.

## Setup IBM federated learning

To run projects in IBM federated learning, you must first install all the requirements. 
We highly recommend using Conda installation for this project. If you don't have Conda,
you can [install it here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

#### Installation with Conda (recommended)

1. If you already have Conda installed, create a new environment for IBM FL by running:

    `conda create -n <env_name> python=3.6`

    **Note**: Lastest IBM FL library supports Keras model training with two different 
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
python examples/generate_configs.py -m keras_classifier -n 2 -d mnist -p examples/data/mnist/random 
```

This command would generate the configs for the `keras_classifier` model, assuming 2 parties.
You must also specify the party data path via `-p`. 

Run `python examples/generate_configs.py -h` for full descriptions of the different options.

## Initiate Learning

#### Start the Aggregator

To start the aggregator, open a terminal window running the IBM FL environment set up previously.

1. In the terminal run:
    ```commandline
    python -m ibmfl.aggregator.aggregator examples/configs/keras_classifier/config_agg.yml
    ```  

    where the path provided is the aggregator config file path.

2. Then in the terminal, type `START` and press enter.

#### Register Parties

To register new parties, open a new terminal window for each party, running the IBM FL environment set up previously.

1. In the terminal run:
     ```commandline
    python -m ibmfl.party.party examples/configs/keras_classifier/config_party0.yml
    ``` 

    where the path provided is the path to the party config file.

    *NOTE*: Each party will have a different config file, usually noted by changing `config_party<idx>.yml`

2.  In the terminal for each party, type `START` and press enter.

3. Then in then terminal for each party, type `REGISTER` and press enter.

#### Train

To initiate federated training, type `TRAIN` in your aggregator terminal and press enter.