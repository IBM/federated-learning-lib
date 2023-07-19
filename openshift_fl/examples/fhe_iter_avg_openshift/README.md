# Running Fully Homomorphic Encryption (FHE) based Secure Aggregation for Iterative Averaging Fusion using OpenShift Cluster

This example explains how to run FHE based secure aggregation with federated learning on CNNs.

This experiment can be run using models with different underlying frameworks.
The following frameworks are supported (for specific datasets as outlined in the next section).

|       Model Type                     |  Params   |
|:------------------------------------:|:--------: |
|   Keras (with tf 1.15)               |  keras    |
|   TensorFlow (with tf > 2)           |  tf       |
|   PyTorch (with torch > 1.10 )       |  pytorch  |
|   Scikit Learn (with sklearn > 0.23) |  sklearn  |

## Dataset Setup

Iterative Avg fusion with FHE-based secure aggregation can be run on different datasets by just changing -d param while generating config. Model definition changes as dataset changes, we currently only support below shown combinations.

|       Dataset      |  Params   |   Keras  |  Pytorch |    TF    |  sklearn |
|:------------------:|:--------: |:--------:|:--------:|:--------:|:--------:|
|        MNIST       |   mnist   |    YES   |    YES   |    YES   |    YES   |
|    Adult Dataset   |   adult   |     NO   |     NO   |     NO   |    YES   |
|     Cifar-10       |  cifar10  |    YES   |     NO   |     NO   |     NO   |
|      FEMNIST       |  femnist  |    YES   |     NO   |     NO   |     NO   |

## Setting up Artifacts for the Experiment

- Ensure you are in the root folder of IBMFL project.
  
- Set up and activate the correct Python environment following the tutorial [here](../../../setup-crypto.md).

- Set the `PYTHONPATH` environment variable:

  ```sh
  export PYTHONPATH=".":$PYTHONPATH
  ```

## Prepare Dataset and Configuration

- Split data sample by running:

    ```sh
    python examples/generate_data.py -n <num_parties> -d <dataset> -pp <points_per_party>
    ```

  - Option `-n <num_parties>` specifies the number of participants in the FL training.
  - Option `-d <dataset>` specifies the dataset used. In this example, we use MNIST dataset, i.e., `-d mnist`.
  - Option `-pp <points_per_party>` specifies the number of data points per participant.

- Generate config files by running:

    ```sh
    python examples/generate_configs.py -n <num_parties> -f fhe_iter_avg_openshift -m <model_name> -d <dataset> -p <data_path> -conf_path <staging_dir_path> -context openshift
    ```

  - Option `-n <num_parties>` specifies the number of participants in the FL training.
  - Option `-f <fusion>` specified the example we are using in the FL training. In this example, the fusion is `-f fhe_iter_avg_openshift`.
  - Option `-m <model_name>` specifies the machine learning model that will be trained in this example.
  - Option `-d <dataset>` specifies the dataset used for training. It should be as same as the dataset prepared in last step.
  - Option `-p <path>` denotes the path of dataset.

  Ensure that the path provided for `-p <data_path>` matches the one where the data files were written to in the data splitting step.

- `staging_dir_path` passed for both split data and generate config steps should be same and should be an **absolute path**. After completion of the data splitting and the config generation steps, the `staging_dir_path` should contain the following folders:

  - `data` - contains party data files for train and test
  - `configs` - contains aggregator config file, party config files and model file
  - `datasets` - source dataset
  - `keys` - FHE client and party keys

  The project context set to openshift , so the examples are run from the openshift folders with the following structure (for a two party run, using MNIST dataset):

    ```txt
    <staging_dir_path>
    ├── configs
    │   └── fhe_iter_avg_openshift
    │       └── keras
    │           ├── compiled_keras.h5
    │           ├── config_agg.yml
    │           ├── config_party0.yml
    │           └── config_party1.yml
    ├── data
    │   └── mnist
    │       └── random
    │           ├── data_party0.npz
    │           └── data_party1.npz
    ├── datasets
    │   └── mnist.npz
    └── keys
        ├── fhe.context
        └── fhe.key
    ```

   This folder structure is required by openshift runner to parse and copy files to aggregator and party pods respectively.

   Default behaviour of orchestrator is to copy dataset and model artifacts to PODS, but if COS (Cloud Object Storage) is used to store datasets and model artifacts, then upload data files (`mnist.npz, data_party0.npz, data_party1.npz`), model files (`compiled_keras.h5`), and keys (`fhe.context`, `fhe.key`) to COS bucket. The last section will describe how to set PVC name in orchestrator config for COS bucket access.

## Build the IBMFL DockerFile Image

You will need access to docker repository like `docker hub` before you can execute the below commands. The below commands assume that you are logged into docker repo (docker hub) using docker cli.

- Build the IBMFL docker image. More details can be found in [docker.md](../../../docker.md) for building the docker image.

  ```sh
  docker build --build-arg BACKEND=keras,crypto -t ffl-base .
  ```

- Tag the docker image

  ```sh
  docker tag ffl-base:[tag] [docker repo URL]/ffl-base:[tag] 
  ```

  Please replace `[docker repo URL]` with docker repository URL and `[tag]` with image version number.
  
- Push image to docker repo

  ```sh
  docker push [docker repo URL]/ffl-base:[tag]
  ```

- Edit `openshift_fl/ibmfl-base.json` file and change `DockerImage` name tag to point to pushed docker image `[docker repo URL]/ffl-base:[tag]` as shown below

  ```json
  "from": {
    "kind": "DockerImage",
    "name": "[docker repo URL]/ffl-base:[tag]"
  }
  ```
  
## Install the IBMFL DockerFile Image in OpenShift Clusters

Refer the instructions on the IBM Cloud page to install and setup the OpenShift CLI. Install the IBMFL image in each of OpenShift Clusters by following the below commands.

- Once you have a cluster setup and listed in the Clusters tab on cloud.ibm.com, navigate to the cluster, Click the `Actions` dropdown from the top right of your screen. Click `Connect via CLI` and follow the instructions on the pop up that shows.

    ```sh
    oc login --token=[token key] --server=[cluster url]
    ```

- Verify your OpenShift credentials are set up correctly to access the cluster, using a command like `oc get pods`.

- Install the IBMFL image to OpenShift Image Streams

    ```sh
    oc apply -f openshift_fl/ibmfl-base.json
    ```

    Use the commands `oc get imagestreams` to view the installed IBMFL images which will start with prefix `ibmfl`.
  
- In case you want to re-install the IBMFL image due to version change, please delete old images which starts with  prefix `ibmfl` using `oc delete imagestream [image_name]` and run `oc apply -f openshift_fl/ibmfl-base.json` again.
  
### Run the IBMFL OpenShift Orchestrator

- Edit orchestrator config `openshift_fl/config_openshift.yml` file keys as follows:

key          | description
------------ | -----------
`kube_config_location`| path to kube config file. Remove key `kube_config_location` from orchestrator config if you want orchestrator to use the default kube config file (~/.kube/config)
`staging_dir`| absolute staging directory path where the keras experiment config and mnist data files are stored i.e.`staging_dir_path`
`context_name`  | context name of Openshift Cluster defined in the kube config file
`namespace`  | namespace of Openshift Cluster defined in the kube config file
`data:pvc_name`| Persistent volume claim name (PVC) that points to the COS bucket which holds the datasets and model artifacts. Remove key `data:pvc_name` key from orchestrator config in case you want the datasets and model artifacts to be copied to PODS rather than using COS bucket.

For multiple openshift clusters, add new entry to `cluster_list` with `context_name` and `namespace` of each cluster.

If you plan to use COS bucket for storage, please set up Persistent Volume (PV) and Persistent Volume Claim (PVC) in OpenShift Cluster and provide the PVC name in `data:pvc_name` key.

Other config keys are set with default values, if you want to modify these keys please refer to `openshift_fl/README.md` documentation.

- Next, in a terminal running an activated FL environment, start the openshift runner by executing:

    ```sh
    python openshift_fl/orchestrator.py openshift_fl/examples/config_openshift.yml
    ```

- Aggregator and Party logs files for experiment trial runs get stored in `staging_dir/[experiment_id]/[trial_num]/logs` folder .
