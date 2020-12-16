# Explore Available Examples

### Training Keras models with different fusion algorithms:
* [Running federated averaging (FedAvg) with Keras](fedavg)
* [Simple average with Keras](keras_classifier)
* [FedPlus with Keras](fedplus)
* [Gradient aggregation with Keras](keras_gradient_aggregation)
* [PFNM with Keras](pfnm)
* [Coordinate median with Keras](coordinate_median)
* [Krum with Keras](krum)
* [Zeno with Keras](zeno)

### Training TensorFlow 2.1 models:
* [TensorFlow 2.1 model (`.pb` format)](tf_classifier)
* [Keras with TensorFlow 2.1 backend (`.h5` format)](tf_keras_classifier)

### Training PyTorch models
* [Running federated averaging (FedAvg) with PyTorch](pytorch_classifier)

### Training scikit-learn models in IBM FL:
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

### Jupyter Notebooks to run IBM FL:
* [Keras Classifier](keras_classifier)
* [Reinforcement learning Cartpole](rl_cartpole)

## IBM FL Command Reference


| IBM FL Command | Participant | Description |
| :-----------: | :-----------: | :----------- |
| `START` | aggregator / party | Start accepting connections|
| `REGISTER` | party | Join an FL project |
| `TRAIN` | aggregator | Initiate training process |
| `SYNC` | aggregator | Synchronize model among parties |
| `STOP` | aggregator | End experiment process |
| `EVAL` | party | Evaluate model |

