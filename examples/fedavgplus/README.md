# Running Federated Averaging Plus (fedavgplus) with TensorFLow and PyTorch

**FedAvg Plus is one variation of Fed+ fusion algorithms proposed here: [Fed+: A Unified Approach to Robust Personalized Federated Learning](https://arxiv.org/pdf/2009.06303.pdf)**

More variations of Fed+ can be at:

1. [Coordinate_median_plus](../coordinate_median_plus)
2. [Geometric_median_plus](../geometric_median_plus)


This example explains how to run Federated Avg plus algorithm on CNNs implemented with TensorFlow training on [MNIST](http://yann.lecun.com/exdb/mnist/) data. Data in this example is preprocessed by scaling down to range from [0, 255] to [0, 1]. No other preprocessing is performed.

## Model Setup

This experiment can be run using models with different underlying framework. By default, configs with keras(tf 1.15) 
based model are generated, but other models like PYTORCH, Scikit Learn, keras(tf 2.1) can be creating by changing -m param.


|       Model Type           |  Params   |
|:--------------------------:|:--------: |
|         Pytorch            |  pytorch  |
|   Tensorflow/keras( tf 2.1) |  tf   |

## Setup FL


- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d mnist -pp <points_per_party>
    ```
For example, to generate sample data on MNIST dataset, you could run:
```
python examples/generate_data.py -n 2 -d mnist -pp 200
```

Run `python examples/generate_data.py -h` for full descriptions
of the different options. 

- Generate config files by running:
    ```
    python examples/generate_configs.py -n <num_parties> -f fedavgplus -m tf -d <dataset> -p <path>
    ```
To run FL, you must have configuration files for the aggregator and for each party.

You can generate these config files using the `generate_configs.py` script.

For example, you could run:

```
python examples/generate_configs.py -f fedavgplus -m tf -n 2 -d mnist -p  examples/data/mnist/random
```

This command would generate the configs for the `tf_classifier_mnist` model, assuming 2 parties.
You must also specify the party data path. 

Run `python examples/generate_configs.py -h` for full descriptions of the different options.

- In a terminal running an activated IBM FL environment 
(refer to Quickstart in our website to learn more about how to set up the running environment), start the aggregator by running:
    ```
    python -m ibmfl.aggregator.aggregator <agg_config>
    ```
    Type `START` and press enter to start accepting connections
- In a terminal running an activated IBM FL environment, start each party by running:
    ```
    python -m ibmfl.party.party <party_config>
    ```
    Type `START` and press enter to start accepting connections.
    
    Type  `REGISTER` and press enter to register the party with the aggregator. 
- Finally, start training by entering `TRAIN` in the aggregator terminal.