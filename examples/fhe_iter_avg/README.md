
# Running Fully Homomorpfic Encryption (FHE) based Secure Aggregation for Iterative Averaging Fusion

Federated Learning inherently provides some level of privacy because parties do not need to share their raw training data. However, some federations have stringent privacy requirements or are subject to regulation that may dictate that the system needs to add additional protection mechanisms to prevent inference of private information.

Homomorphic Encryption can help us reduce the risk by hiding the final model from the aggregator and only revealing the aggregated version to the parties. Homomorphic Encryption is a crypto system that allows an entity to perform computations on encrypted data without decrypting it. In other words, it enables the computation of a function over encrypted inputs and produces the result in encrypted form.

IBM FL uses the *[Cheon-Kim-Kim-Song (CKKS) scheme](https://eprint.iacr.org/2016/421.pdf)* for Homomorphic Encryption. HE functionalities are implemented using *[IBM HElayers software development kit (SDK)](https://github.com/IBM/helayers)*, and in particular, its *[PyHElayers](https://github.com/IBM/helayers#pyhelayers-python-package)* Python package. You can install `pyhelayers` in your conda environment by running `pip install pyhelayers`. Note that `pyhelayers` is currently supported only on Linux (x86 and IBM Z).

After enabling IBM FL with HE, the aggregator does not see model updates in plaintext. The aggregator sees encrypted model updates and performs aggregation under encryption.

This example explains how to run FHE based secure aggregation with federated learning on CNNs. 


## Model Setup

This experiment can be run using models with different underlying frameworks. 
The following frameworks are supported (for specific datasets as outlined in the next section).

|       Model Type                     |  Params   |
|:------------------------------------:|:--------: |
|   Keras (with tf 1.15)               |  keras    |
|   TensorFlow (with tf > 2)           |  tf       |
|   PyTorch (with torch > 1.10 )       |  pytorch  |
|   Scikit Learn (with sklearn > 0.23) |  sklearn  |

## Dataset Setup
Iterative Avg fusion with FHE-based secure aggregation can be run on different datasets by just changing -d param while generating config. Model definition changes as dataset changes, we currently only support below shown combinations.

|       Dataset      |  Params   |   Keras  |  Pytorch |    TF    |  sklearn |
|:------------------:|:--------: |:--------:|:--------:|:--------:|:--------:|
|        MNIST       |   mnist   |    YES   |    YES   |    YES   |    YES   |
|    Adult Dataset   |   adult   |     NO   |     NO   |     NO   |    YES   | 
|     Cifar-10       |  cifar10  |    YES   |     NO   |     NO   |     NO   |
|      FEMNIST       |  femnist  |    YES   |     NO   |     NO   |     NO   |




**Instructions to Run this Example** 

***Prepare Dataset and Configuration***

- Split data sample by running:

    ```
    python examples/generate_data.py -n <num_parties> -d <dataset> -pp <points_per_party>
    ```
    + Option `-n <num_parties>` specifies the number of participants in the FL training.
    + Option `-d <dataset>` specifies the dataset used. In this example, we use MNIST dataset, i.e., `-d mnist`.
    + Option `-pp <points_per_party>` specifies the number of data points per participant.


- Generate config files by running:
    ```bash
    python examples/generate_configs.py -n <num_parties> -f fhe_iter_avg -d <dataset> -p <path>
    ```

    + Option `-n <num_parties>` specifies the number of participants in the FL training.
    + Option `-f <fusion>` specified the example we are using in the FL training. In this example, the fusion is `-f fhe_iter_avg`.
    + Option `-m <model_name>` specifies the machine learning model that will be trained in this example. 
    + Option `-d <dataset>` specifies the dataset used for training. It should be as same as the dataset prepared in last step.
    + Option `-p <path>` denotes the path of dataset.

   
***Training Instructions***

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