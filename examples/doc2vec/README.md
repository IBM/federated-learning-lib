# Running Doc2VecFLModel instance from a Gensim Doc2vec model

**Doc2vec proposed in: [SEEC: Semantic vector federation across edge computing environments](https://arxiv.org/pdf/2008.13298.pdf)**

This example explains how to run the Doc2Vec algorithm with Gensim
on [Wikipedia](https://radimrehurek.com/gensim/wiki.html) data.

## Model Setup

This experiment can be run using Gensim Doc2vec model.


|       Model Type           |  Params   |
|:--------------------------:|:--------: |
|      Doc2vec model         |  gensim   |


## Setup


- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d wikipedia -pp <points_per_party>
    ```
- Generate config files by running:
    ```
    python examples/generate_configs.py -n <num_parties> -f doc2vec -m doc2vec -d wikipedia -p <path>
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
