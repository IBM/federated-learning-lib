

# Running SPAHM for unsupervised clustering problem with KMeans

**SPAHM proposed here: [Statistical Model Aggregation via Parameter Matching](https://arxiv.org/abs/1911.00218)**

This example explains how to run the SPAHM algorithm on a simulated federated unsupervised clustering dataset using 
a KMeans model. For more details on the dataset generation process and detailed experimental results 
of SPAHM on this problem, please refer to 
[Section 5 in the SPAHM paper for details](https://papers.nips.cc/paper/9277-statistical-model-aggregation-via-parameter-matching.pdf).


- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d federated-clustering -pp <points_per_party>
    ```
- Generate config files by running:
    ```
    python examples/generate_configs.py -n <num_parties> -f spahm -m sklearn -d federated-clustering -p <path>
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