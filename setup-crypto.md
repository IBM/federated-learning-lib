# Setup with Crypto

This tutorial explains how to setup with crypto and run IBM federated learning using fully homomorphic encryption (HE). All commands are assumed to be run from the base directory at the top of this repository.

**Note:** This will only work on a Linux machine (x86 and IBM Z). Other operating systems are not supported.

## Setup IBM federated learning

To run projects in IBM federated learning, you must first install all the requirements. We highly recommend using Conda installation for this project. If you don't have Conda, you can [install it here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

The latest IBM FL library supports model training using Keras (with TensorFlow v1), TensorFlow v2, PyTorch, and Scikit-learn. It is recommended to install IBM FL in different conda environments for the Keras and TensorFlow v2 versions. Models using PyTorch or Scikit-learn will work on either.

### Installation with Conda (recommended)

1. If you already have Conda installed, create a new environment for IBM FL. We recommend using Python 3.6, but newer versions may also work.

    a. If running experiments using Keras models (with Tensorflow v1), create a new environment by running:

    ```bash
    conda create -n <env_name> python=3.6 tensorflow=1.15
    ```

    b. If running experiments using TensorFlow v2, create a new environment by running:

    ```bash
    conda create -n <env_name> python=3.6
    ```

    c. If running experiments using PyTorch or Scikit-learn, either environment will work.

2. Activate the new Conda environment by running:

    ```bash
    conda activate <env_name>
    ```

    If using TensorFlow v2, install the package:

    ```bash
    pip install tensorflow==2.1.0
    ```

    If this version of TensorFlow is unavailable, try installing a newer version.

3. Install the necessary packages needed for IBM HElayers to use fully homomorphic encryption:

    ```bash
    pip install cryptography pyhelayers
    ```

4. Install the IBM FL package by running:

    ```bash
    pip install <IBM_federated_learning_whl_file>
    ```

### Installation with virtualenv

We recommend using Python 3.6, but newer versions may also work.

1. Create a virtual environment by running:

    ```bash
    python -m pip install --user virtualenv
    virtualenv venv
    source venv/bin/activate
    python -m pip install --upgrade pip
    ```

2. Install basic dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    If using TensorFlow v2, install the package:

    ```bash
    pip install tensorflow==2.1.0
    ```

    If this version of TensorFlow is unavailable, try installing a newer version.

3. Install the necessary packages needed for IBM HElayers to use fully homomorphic encryption:

    ```bash
    pip install cryptography pyhelayers
    ```

4. Install the IBM FL package by running:

    ```bash
    pip install <IBM_federated_learning_whl_file>
    ```

## Run Tutorial Notebook

Follow one of the [tutorial notebooks](Notebooks) for getting starting with crypto and running IBM federated learning using fully homomorphic encryption (HE). Tutorials currently exist for:

* (Optional) [Generating homomorphic encryption key files](Notebooks/crypto_fhe_key_setup.ipynb)
* [Training a TensorFlow model using homomorphic encryption](Notebooks/crypto_fhe_tensorflow)
* [Training a PyTorch model using homomorphic encryption](Notebooks/crypto_fhe_pytorch)
* [Training a Scikit-learn model using homomorphic encryption](Notebooks/crypto_fhe_sklearn)
