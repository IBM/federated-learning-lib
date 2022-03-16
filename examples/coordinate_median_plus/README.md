# Running Coordinate Median Plus (coordinate_median_plus) with TensorFLow

**Coordinate Median Plus is one variation of Fed+ fusion algorithms proposed here: [Fed+: A Unified Approach to Robust Personalized Federated Learning](https://arxiv.org/pdf/2009.06303.pdf)**


More variations of Fed+ can be at:

1. [Fedavg_plus](../fedavgplus)
2. [Geometric_median_plus](../geometric_median_plus)

This example explains how to run coordinate median plus algorithm on CNNs implemented with TensorFlow training on
[MNIST](http://yann.lecun.com/exdb/mnist/) data. Data in this example is preprocessed by scaling down to range from `[0, 255]` to `[0, 1]`.
No other preprocessing is performed.

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
    python examples/generate_configs.py -n <num_parties> -f coordinate_median_plus -m tf -d <dataset> -p <path>
    ```

To run FL, you must have configuration files for the aggregator and for each party.

You can generate these config files using the `generate_configs.py` script.
 
For example, you could run:

```
python examples/generate_configs.py -f coordinate_median_plus -m tf -n 2 -d mnist -p  examples/data/mnist/random
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