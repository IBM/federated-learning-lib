# Explore Available Examples

### Training Keras models with different fusion algorithms:
* [Running Federated Averaging with Keras](fedavg)
* [Simple average with Keras](keras_classifier)
* [Gradient aggregation with Keras](keras_gradient_aggregation)
* [PFNM with Keras](pfnm)
* [Coordinate median with Keras](coordinate_median)
* [Krum with Keras](krum)
* [Zeno with Keras](zeno)

### Training scikit-learn models in IBM FL:
* [SGDClassifier](sklearn_sgdclassifier)
* [Logistic classifier](sklearn_logclassification)
* [SPAHM with KMeans](spahm)
* [Differential private Naive Bayes models](naive_bayes)

### ID3 Decision trees:
* [Training ID3 decision trees](id3_dt)

### Reinforcement learning in IBM FL:
* [Cartpole](rl_cartpole)
* [Pendulum](rl_pendulum)

### Unsupervised learning in IBM FL:
* [SPAHM with KMeans](spahm)


## IBM FL Command Reference


| IBM FL Command | Participant | Description |
| :-----------: | :-----------: | :----------- |
| `START` | aggregator / party | Start accepting connections|
| `REGISTER` | party | Join an FL project |
| `TRAIN` | aggregator | Initiate training process |
| `SYNC` | aggregator | Synchronize model among parties |
| `STOP` | aggregator | End experiment process |
| `EVAL` | party | Evaluate model |

