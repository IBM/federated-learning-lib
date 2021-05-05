
# Running FedProx with Tensorflow

**Note that it requires tensorflow version 2.1.0**

FedProx is an optimization framework that tackles the systems and statistical heterogeneity in federated networks. 
FedProx allows for variable amounts of work to be performed locally across devices, and relies on a proximal term to help stabilize
the method. 

FedProx proposed here: [FEDERATED OPTIMIZATION IN HETEROGENEOUS NETWORKS](https://arxiv.org/pdf/1812.06127.pdf)

This example explains how to run FedProx algorithm on [CNNs implemented with Tensorflow](https://www.tensorflow.org/tutorials/quickstart/advanced) training on
[MNIST](http://yann.lecun.com/exdb/mnist/) data using FedProx Optimizer - Perturbed Gradient Descent. 
Data in this example is preprocessed by scaling down to range from `[0, 255]` to `[0, 1]`.
No other preprocessing is performed.

- Set FL environment by running:

    ```
    ./setup_environment.sh <env_name>  # setup environment
    ```
- Activate a new FL environment by running:

    ```
    conda activate <env_name>          # activate environment
    ```
- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d mnist -pp <points_per_party>
    ```
- Generate config files by running:
    ```
    python examples/generate_configs.py -f fedprox -m tf -d mnist -n <num_parties> -p <path>
    ```
- In a terminal running an activated FL environment, start the aggregator by running:
    ```
    python -m ibmfl.aggregator.aggregator <agg_config>
    ```
    Type `START` and press enter to start accepting connections
- In a terminal running an activated FL environment, start each party by running:
    ```
    python -m ibmfl.party.party <party_config>
    ```
    Type `START` and press enter to start accepting connections.
    
    Type  `REGISTER` and press enter to register the party with the aggregator. 
- Finally, start training by entering `TRAIN` in the aggregator terminal.
