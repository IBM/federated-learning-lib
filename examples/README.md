# Explore Available Examples

### Training Keras, TensorFlow 2.1 and PyTorch models with different fusion algorithms:
* [Running federated averaging (FedAvg)](fedavg)
* [Simple average](iter_avg)
* [Shuffle iterative average](shuffle_iter_avg)
* [FedAvgPlus with Tensorflow and PyTorch](fedavgplus)
* [Gradient aggregation](gradient_aggregation)
* [PFNM with Keras](pfnm)
* [Coordinate median](coordinate_median)
* [Coordinate median plus](coordinate_median_plus)
* [Geometric median plus](geometric_median_plus)
* [Krum with Keras](krum)
* [Zeno with Keras](zeno)

### Training scikit-learn models in IBM FL:
* [Logistic classifier](iter_avg)
* [SPAHM with KMeans](spahm)
* [Differential private Naive Bayes models](naive_bayes_dp)

### ID3 Decision trees:
* [Training ID3 decision trees](id3_dt)

### Reinforcement learning in IBM FL:
* [Cartpole](rl_cartpole)
* [Pendulum](rl_pendulum)

### Unsupervised learning in IBM FL:
* [SPAHM with KMeans](spahm)

### Jupyter Notebooks to run IBM FL:
* [Keras Classifier](../Notebooks/keras_mnist_classifier)
* [Reinforcement learning Cartpole](../Notebooks/reinforcement_learning_cartpole)

## IBM FL Command Reference


| IBM FL Command | Participant | Description |
| :-----------: | :-----------: | :----------- |
| `START` | aggregator / party | Start accepting connections|
| `REGISTER` | party | Join an FL project |
| `TRAIN` | aggregator | Initiate training process |
| `SYNC` | aggregator | Synchronize model among parties |
| `STOP` | aggregator | End experiment process |
| `EVAL` | party | Evaluate model |

