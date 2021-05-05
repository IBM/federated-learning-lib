# Running differential private simple average with Sklearn Naive Bayes models

This example explains how to run the simple averaging fusion algorithm to train a Naive Nayes model implemented with Scikit-learn 
on [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/Adult) data. 
The following preprocessing was performed in `AdultSklearnDataHandler` on the original dataset:
  * Drop following features: `workclass`, `fnlwgt`, `education`, `marital-status`, `occupation`, `relationship`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`
  * Map `race`, `sex` and `class` values to 0/1
  * Split `age` and `education` columns into multiple columns based on value

No other preprocessing is performed.

- Split data by running:
    
    ```
    python examples/generate_data.py -n <num_parties> -d adult -pp <points_per_party>
    ```
- Generate config files by running:
    ```
    python examples/generate_configs.py -n <num_parties> -f naive_bayes_dp -m sklearn -d adult -p <path>
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