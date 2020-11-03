# Quorum handling and ability to Rejoin

## Quorum handling
IBM FL supports the functionality to specify quorum percentage in the aggregator config file to provide flexibility to parties that have potential connectivity failure. Given a total number of parties registered at a particular round, the quorum percentage defines the minimum number of parties that should reply back for that round. If for some round aggregator receives less number of replies from the parties, it will stop the federated learning process. This functionality makes sure that if for some reasons a number of parties dropout they can rejoin back as long as the available parties do not fall below the quorum value.

For example in following configuration file `perc_quorum` is set to 0.75. This means that for each round aggregator will expect 75% of the registered parties to reply back. So if there are 20 parties that registered, federated learning will continue as long as not more than five parties drop out. 

```
hyperparams:
  global:
    max_timeout: 60
    num_parties: 5
    perc_quorum: 0.75
    rounds: 3
    termination_accuracy: 0.9
```

## Maximum Timeout and Rejoin
Users can specify the maximum timeout (in seconds) aggregator should wait for parties to reply back in the aggregator configuration file. If `max_timeout` value is specified, aggregator will wait for specified amount of time to check if the required number of parties (calculated based on the quorum percentage provided earlier) have replied back or not. Please note that if quorum percentage is not specified aggregator will expect the value to be 100% and expect reply from all the registered parties. Similarly, if maximum timeout is not specified aggregator will wait forever for parties to reply back.

To rejoin party just needs to issue START and REGISTER commands like it did initially to join federated learning process.
