
# Running Krum with Keras

**Krum proposed in: [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf)**

This example explains how to run the Krum fusion algorithm to train CNNs implemented with Keras training
on [MNIST](http://yann.lecun.com/exdb/mnist/) data. Data in this example preprocessed by scaling down to range from `[0, 255]` to `[0, 1]`.
No other preprocessing is performed.

**Note that** the parameter `byzantine_threshold` is set to `1` by default. According to Krum's assumption, the number of parties in the FL system should be at least `2 * byzantine_threshold + 3`. Therefore, to run this example, please create at least `5` parties.

- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d mnist -pp <points_per_party>
    ```
- Generate config files by running:
    ```
    python examples/generate_configs.py -n <num_parties> -m krum   -d mnist -p <path>
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