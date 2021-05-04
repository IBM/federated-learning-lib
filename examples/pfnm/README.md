
# Running Probabilistic Federated Neural Matching (PFNM) with Keras

**PFNM proposed here: [Bayesian Nonparametric Federated Learning of Neural Networks](https://arxiv.org/abs/1905.12022)**

This example explains how to run the PFNM algorithm on multi-layer
perceptrons (fully-connected networks) implemented with Keras training on
[MNIST](http://yann.lecun.com/exdb/mnist/) data. Data in this example is
preprocessed by scaling down to range from `[0, 255]` to `[0, 1]`.
No other preprocessing is performed.


## Model Setup

This experiment can be run using models with different underlying framework. By default, configs with keras(tf 1.15) 
based model are generated, but other models like PYTORCH, Keras(tf 2.1) can be creating by changing -m param.


|       Model Type             |  Params   |
|:----------------------------:|:--------: |
|   Keras (with tf 1.15)       |  keras    |
|         Pytorch              |  pytorch  |
|   Tensorflow/keras( tf 2.1)  |    tf     |


- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d mnist -pp <points_per_party>
    ```
- Generate config files by running:
    ```
    python examples/generate_configs.py -f pfnm -m keras -d mnist -n <num_parties> -p <path>
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