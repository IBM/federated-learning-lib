
# Running Comparative Elimination with Tensorflow

**Comparative Elimination (CE) proposed in: [Byzantine Fault-Tolerance in Federated Local SGD under 2f-Redundancy](https://arxiv.org/abs/2108.11769)**

This example explains how to run the CE fusion algorithm to train CNNs implemented with Tensorflow/Keras training
on [MNIST](http://yann.lecun.com/exdb/mnist/) data. Data in this example is preprocessed by scaling down to range from `[0, 255]` to `[0, 1]`.
No other preprocessing is performed.
## Model Setup

This experiment can be run using models with different underlying framework. By default, configs with Keras (tf 2.4.1) based model are generated, but other models like PyTorch, Keras(tf 1.15) can be creating by changing -m param.


|       Model Type             |  Params   |
|:----------------------------:|:--------: |
|   Keras (with tf 1.15)       |  keras    |
|         Pytorch              |  pytorch  |
|   Tensorflow/Keras(tf 2.4.1) |  tf       |


**Note that** the parameter `byzantine_threshold` is set to `1` by default. According to CE's assumptions, the number of parties in the FL system should be at least `2 * byzantine_threshold + 1`. Therefore, to run this example, please create at least `3` parties.

- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d mnist -pp <points_per_party>
    ```
- Generate config files by running:
    ```
    python examples/generate_configs.py -f comparative_elimination -m tf -n <num_parties> -d mnist -p <path>
    ```
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
