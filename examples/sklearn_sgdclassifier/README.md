
# Running Scikitlearn SGD Classifier in IBM federated learning

This example explains how to run federated learning on an SGD Classifier, implemented with Scikit-Learn
training on [MNIST](http://yann.lecun.com/exdb/mnist/) data. The following preprocessing was performed on the original dataset:

* Data is scaled down to range from `[0, 255]` to `[0, 1]`
* Images are reshaped from`[28, 28]` to `[1,784]`

No other preprocessing is performed.

- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d mnist -pp <points_per_party>
    ```
- Generate config files by running:
    ```
    python examples/generate_configs.py -n <num_parties> -m sklearn_sgdclassifier -d mnist -p <path>
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