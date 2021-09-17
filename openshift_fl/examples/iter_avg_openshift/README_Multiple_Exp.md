
# Running Multiple Experiments - Keras and PyTorch CNN in FL using OpenShift Cluster

This example explains how to run federated learning on CNNs implemented with Keras and Pytorch training on
[MNIST](http://yann.lecun.com/exdb/mnist/) data using OpenShift cluster. Data in this example is preprocessed by scaling down to range from `[0, 255]` to `[0, 1]`.
No other preprocessing is performed.

### Setting up artifacts for the experiment
- Set up the correct FL environment following our tutorial [here](https://github.com/IBM/federated-learning-lib/blob/main/quickstart.md#1-set-up-a-running-environment-for-ibm-federated-learning). 

- Activate a new FL environment by running:

    ```
    conda activate <env_name>          # activate environment
    ```

Next we set up the Keras and Pytorch experiments, Please ensure that `staging_dir_path_1` and `staging_dir_path_2` are seperate directories.

### Setting up Keras MNIST experiment

- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d mnist -pp <points_per_party> -p <staging_dir_path_1>
    ```
- Generate config files by running:
    ```
    python examples/generate_configs.py -f iter_avg_openshift -m keras -n <num_parties> -d mnist -p <data_path> -conf_path <staging_dir_path_1> -context openshift
    ```
    Ensure that the path provided for `-p <data_path>` matches the one where the data files were written to in the data splitting step.
    
- `staging_dir_path_1` passed for both split data and generate config steps should be same and should be an **absolute path**. After completion of the data splitting and the config generation steps, the `staging_dir_path_1` should contain the following folders:
   
   `data` - contains party data files for train and test
   
   `configs` - contains aggregator config file, party config files and model file
   
   `datasets` - source dataset 

   `context` - project context set to openshift , so the examples are run from the openshift folders 

    with the following structure (for a two party run, using MNIST dataset):


    ```
    <staging_dir_path_1>/
    ├── configs
    │   └── iter_avg_openshift
    │         └── keras
    │               ├── compiled_keras.h5
    │               ├── config_agg.yml
    │               ├── config_party0.yml
    │               └── config_party1.yml
    ├── data
    │   └── mnist
    │       └── random
    │           ├── data_party0.npz
    │           └── data_party1.npz
    └── datasets
        └── mnist.npz

    ```
   This folder structure is required by openshift runner to parse and copy files to aggregator and party pods respectively.
   
Default behaviour of orchestrator is to copy dataset and model artifacts to PODS, but if COS (Cloud Object Storage) is used to store datasets and model artifacts, then upload data files like `mnist.npz, data_party0.npz, data_party1.npz` and model file like `compiled_keras.h5` to COS bucket. The last section will describe how to set PVC name in orchestrator config for COS bucket access.
### Setting up PyTorch MNIST experiment

- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d mnist -pp <points_per_party> -p <staging_dir_path_2>
    ```
    This step can be skipped, if you want to use data files generated for keras experiment. In this case, copy the data files and datasets to `staging_dir_path_2`.
    
- Generate config files by running:
    ```
    python examples/generate_configs.py -f iter_avg_openshift -m pytorch -n <num_parties> -d mnist -p <data_path> -conf_path <staging_dir_path_2> -context openshift
    ```
    Ensure that the path provided for `-p <data_path>` matches the one where the data files were written to in the data splitting step.
    
- `staging_dir_path_2` passed for both split data and generate config steps should be same and should be an **absolute path**. After completion of the data splitting and the config generation steps, the `staging_dir_path_2` should contain the following folders:
   
   `data` - contains party data files for train and test
   
   `configs` - contains aggregator config file, party config files and model file
   
   `datasets` - source dataset 

    with the following structure (for a two party run, using MNIST dataset):


    ```
    <staging_dir_path_2>/
    ├── configs
    │   └── iter_avg_openshift
    │         └── pytorch
    │               ├── pytorch_sequence.pt
    │               ├── config_agg.yml
    │               ├── config_party0.yml
    │               └── config_party1.yml
    ├── data
    │   └── mnist
    │       └── random
    │           ├── data_party0.npz
    │           └── data_party1.npz
    └── datasets
        └── mnist.npz

    ```
   This folder structure is required by openshift runner to parse and copy files to aggregator and party pods respectively.
   
  Default behaviour of orchestrator is to copy dataset and model artifacts to PODS, but if COS (Cloud Object Storage) is used to store datasets and model artifacts, then upload data files like `mnist.npz, data_party0.npz, data_party1.npz` and model file like `pytorch_sequence.pt` to COS bucket. The last section will describe how to set PVC name in orchestrator config for COS bucket access.

### Build the IBMFL DockerFile Image

You will need access to docker repository like `docker hub` before you can execute the below commands. The below commands assume that you are logged into docker repo (docker hub) using docker cli.

- Build the IBMFL docker image 
  ```
  docker build -t ffl-base .
  ```
- Tag the docker image
  ```
  docker tag ffl-base:[tag] [docker repo URL]/ffl-base:[tag] 
  ```
  Please replace `[docker repo URL]` with docker repository URL and `[tag]` with image version number.
  
- Push image to docker repo
  ```
  docker push [docker repo URL]/ffl-base:[tag]
  ```
- Edit `openshift_fl/ibmfl-base.json` file and change `DockerImage` name tag to point to pushed docker image `[docker repo URL]/ffl-base:[tag]` as shown below
  ```
  "from": {
    "kind": "DockerImage",
    "name": "[docker repo URL]/ffl-base:[tag]"
  }
  ```
  
### Install the IBMFL DockerFile Image in OpenShift Clusters

Refer the instructions on the IBM Cloud page to install and setup the OpenShift CLI. Install the IBMFL image in each of OpenShift Clusters by following the below commands.
- Once you have a cluster setup and listed in the Clusters tab on cloud.ibm.com, navigate to the cluster, Click the `Actions` dropdown from the top right of your screen. Click `Connect via CLI` and follow the instructions on the pop up that shows.

    ```
   oc login --token=[token key] --server=[cluster url]
    ```
    
- Verify your OpenShift credentials are set up correctly to access the cluster, using a command like `oc get pods`.

- Install the IBMFL image to OpenShift Image Streams

    ```
    oc apply -f openshift_fl/ibmfl-base.json
    ```
    Use the commands `oc get imagestreams` to view the installed IBMFL images which will start with prefix `ibmfl`.
  
- In case you want to re-install the IBMFL image due to version change, please delete old images which starts with  prefix `ibmfl` using `oc delete imagestream [image_name]` and run `oc apply -f openshift_fl/ibmfl-base.json` again.
  
### Run the IBMFL OpenShift Orchestrator

- Edit `openshift_fl/config_openshift_multiple_exp.yml` file keys as follows :- 

key           | description
--------------| -----------
`kube_config_location`| path to kube config file. Remove key `kube_config_location` from orchestrator config if you want orchestrator to use the default kube config file (~/.kube/config)
`staging_dir` | Absolute staging directory path where the experiment config and mnist data files are stored, i.e.`staging_dir_path_1` and `staging_dir_path_2` for kerasexp and pytorchexp respectively. 
`context_name`  | context name of Openshift Cluster defined in the kube config file
`namespace`  | namespace of Openshift Cluster defined in the kube config file
`data:pvc_name`| Persistent volume claim name (PVC) that points to the COS bucket which holds the datasets and model artifacts. Remove key `data:pvc_name` key from orchestrator config in case you want the datasets and model artifacts to be copied to PODS rather than using COS bucket.

For multiple openshift cluster, add new entry to `cluster_list` with `context_name` and `namespace` of the cluster.

If you plan to use COS bucket for storage, please set up Persistent Volume (PV) and Persistent Volume Claim (PVC) in OpenShift Cluster and provide the PVC name in `data:pvc_name` key.

Other config keys are set with default values, if you want to modify these keys please refer to `openshift_fl/README.md` documentation.

- Next, in a terminal running an activated FL environment, start the openshift runner by executing:
    ```
    python openshift_fl/orchestrator.py openshift_fl/examples/iter_avg_openshift/config_openshift_multiple_exp.yml
    ```
    
- Aggregator and Party logs files for experiment trial runs get stored in `staging_dir_path_1/[experiment_id]/[trial_num]/logs` and `staging_dir_path_2/[experiment_id]/[trial_num]/logs` folders respectively.
