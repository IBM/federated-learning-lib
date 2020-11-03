
# Running Scikitlearn Logistic Classifier in IBM federated learning

Currently, for logistic classifier we support the following datasets:

* [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/Adult)
* [MNIST](http://yann.lecun.com/exdb/mnist/)


The following preprocessing was performed in `AdultSklearnDataHandler` on the original dataset:
  * Drop following features: `workclass`, `fnlwgt`, `education`, `marital-status`, `occupation`, `relationship`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`
  * Map `race`, `sex` and `class` values to 0/1
  * Split `age` and `education` columns into multiple columns based on value

  Further details in documentation of `preprocess()` in `AdultSklearnDataHandler`.

No other preprocessing is performed.

The following preprocessing was performed on the MNIST dataset:

* Data is scaled down to range from `[0, 255]` to `[0, 1]`
* Images are reshaped from`[28, 28]` to `[1,784]`


No other preprocessing is performed.

- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d <dataset_name> -pp <points_per_party>
    ```
- Generate config files by running:
    ```
    python examples/generate_configs.py -n <num_parties> -m sklearn_logclassification -d <dataset_name> -p <path>
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