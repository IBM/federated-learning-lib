
# Running PENDULUM in IBM federated learning

This example explains how to run federated learning on Pendulum problem
[PENDULUM](https://gym.openai.com/envs/Pendulum-v0/) using RLlib.

- Generate config files by running:
    ```
    python examples/generate_configs.py -n <num_parties> -f rl_pendulum  -p '' -d 'default'
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