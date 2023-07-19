# Setup with Crypto

This tutorial explains how to setup and run IBM federated learning with crypto using fully homomorphic encryption (HE). All commands are assumed to be run from the base directory at the top of this repository.

**Note:** This will only work on a Linux machine (x86 and IBM Z). Other operating systems and architectures are not supported.

## Setup IBM federated learning with crypto

To run projects in IBM federated learning, you must first create a Python environment to install all the requirements. You can use either Conda or venv:

<details>
<summary>Conda (recommended)</summary>

If you don't have Conda, you can install it [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install).

Once installed, create a new Conda environment. We recommend using Python 3.7, but newer versions may also work.

```sh
conda create -n <env_name> python=3.7
```

Activate the newly created Conda environment.

```sh
conda activate <env_name>
```

</details>

<details>
<summary>Venv</summary>

Create a new virtual environment using Python's built-in `venv` module. This will use your system's Python version which may or may not be fully compatible.

```bash
python -m venv venv
```

Activate the newly created virtual environment.

```sh
source venv/bin/activate
```

</details>

After creating and activating the Python environment, install the wheel file to install the IBM federated learning library and all dependencies. This file is located in the `federated-learning-lib` directory of this repo. By default, the wheel file will not install any additional machine learning or crypto libraries. You must specify `crypto` and the desired model training backend library in brackets. The following backends are supported:

```sh
# Install with crypto and no additional machine learning libraries
pip install "/path/to/federated_learning_lib.whl[crypto]"
# Install with crypto and Scikit-learn backend
pip install "/path/to/federated_learning_lib.whl[crypto,sklearn]"
# Install with crypto and PyTorch backend
pip install "/path/to/federated_learning_lib.whl[crypto,pytorch]"
# Install with crypto and Keras (TensorFlow v1) backend
pip install "/path/to/federated_learning_lib.whl[crypto,keras]"
# Install with crypto and TensorFlow v2 backend
pip install "/path/to/federated_learning_lib.whl[crypto,tf]"
```

You can also install multiple backends using a comma separated list. For example:

```sh
# Install with crypto and Scikit-learn and Keras (TensorFlow v1) backend
pip install "/path/to/federated_learning_lib.whl[crypto,sklearn,keras]"
# Install with crypto and PyTorch and TensorFlow v2 backend
pip install "/path/to/federated_learning_lib.whl[crypto,pytorch,tf]"
# Install with crypto and Scikit-learn, PyTorch, and Keras (TensorFlow v1) backend
pip install "/path/to/federated_learning_lib.whl[crypto,sklearn,pytorch,keras]"
```

You may install as many backends as you'd like. The only exception is that you **cannot** install both the Keras and TensorFlow v2 backends since they are different versions of TensorFlow.

**Notes:**

* The quotes are required if using the Zsh shell (this is the default shell for Mac).
* There should be no spaces before or after each comma.
* The Keras backend will only work for Python 3.7.

## Run Tutorial Notebook

Follow one of the [tutorial notebooks](Notebooks) for getting starting with crypto and running IBM federated learning using fully homomorphic encryption (HE). Tutorials currently exist for:

* (Optional) [Generating homomorphic encryption key files](notebooks/crypto_fhe_key_setup.ipynb)
* [Training a TensorFlow model using homomorphic encryption](notebooks/crypto_fhe_tensorflow)
* [Training a PyTorch model using homomorphic encryption](notebooks/crypto_fhe_pytorch)
* [Training a Scikit-learn model using homomorphic encryption](notebooks/crypto_fhe_sklearn)
