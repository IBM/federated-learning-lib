# Running Federated Plus Averaging (fedplus) with Keras

**Fedplus proposed here: [Fed+: A Family of Fusion Algorithms for Federated Learning](https://arxiv.org/pdf/2009.06303.pdf)**

Fedplus supports the following fusion handlers:
1. Iterative average (iter_avg_fusion_handler.py)
2. Coordinate median (coordinate_median_fusion_handler.py)

This example explains how to run the fedplus averaging algorithm on CNNs implemented with Keras training on LEAF FEMNIST. In this 
example we are using Iterative Average fusion handler for learning. Please modify the config accordingly to use Coordinate median. 
All commands are assumed to be run from the base `FL/` directory at the top of this repository.

## Setup FL

To run projects in FL, you must first install all the requirements. 
We highly recommend using Conda installation for this project. If you don't have Conda,
you can [install it here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

- Split data by running:

You can use `generate_data.py` to generate sample data on any of FL's integrated datasets. For example, you could run:
```
python examples/generate_data.py -n 2 -d femnist -pp -1
```
Note: This might take some time to download and unpack the contents.

LEAF FEMNIST has its own preset data distribution for each client. To use this distribution, we set the -pp to -1. If we
set the -pp to a different number, it will be using stratified non-iid data distribution. In order to use IID distribution
set --stratify to False.
This above command would generate 2 parties with the default distributions. The LEAF paper (https://arxiv.org/abs/1812.01097)
uses around 200 clients for demonstrating FEMNIST. For more information on what preprocessing was performed, 
check the [Keras classifier example](keras_classifier_femnist).

Run `python examples/generate_data.py -h` for full descriptions
of the different options. 

- Generate config files by running:

```
python examples/generate_configs.py -f fedplus -m keras -n 2 -d femnist -p  examples/data/femnist/orig_dist
```

This command would generate the configs for the `keras` model with with (fedplus)[https://arxiv.org/pdf/2009.06303.pdf] fusion algorithm, assuming 2 parties.
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
