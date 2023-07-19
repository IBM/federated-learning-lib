# IBMFL Multi-Cloud OpenShift Orchestrator

IBMFL Multi-Cloud and Hybrid Cloud Orchestrator automates the deployment and monitoring of aggregator and party process using federated learning library docker image on OpenShift clusters which are setup on different cloud data center regions. IBMFL OpenShift Orchestrator works on either single or multi clusters.

## OpenShift Orchestrator Features

- Authentication using kubeconfig file to multiple OpenShift Clusters
- Creation and deployment of aggregator and party pods to OpenShift Clusters
- Network communication between aggregator and party process using OpenShift Routes
- Runs experiment by coordinating execution of training commands between aggregator and party process
- Support for multiple trials and parallelization of multiple experiments
- Logs and metrics capture for experiments

## OpenShift Orchestrator usage

Please refer to `examples/iter_avg_openshift` folder on how to use IBMFL OpenShift Orchestrator to run IBMFL experiments. In the example, follow instructions in `README.md` and `README_Multiple_Exp.md` to run single and multiple federated experiments respectively.
To run orchestrator on your laptop, you can install the OpenShift community edition like OKD (<https://www.okd.io/>) or minshift (<https://www.okd.io/minishift/>).

## OpenShift Orchestrator Config Description

Please refer to `config_openshift_sample.yml` file for various keys of config file. The config file consists of two main sections: `cluster` and `experiments`.

- `cluster` contains the openshift cluster configuration for aggregator and party pods.

key          | description | mandatory|default_value
------------ | ----------- |----------|------------
`agg_pod:cpu`  | cpu for aggregator pod | no | 2
`agg_pod:mem`| memory for  aggregator pod | no | 4Gi
`party_pod:cpu`  | cpu for party pod | no | 2
`party_pod:mem`| memory for  party pod | no | 4Gi
`kube_config_location`| location of kube config file | no | ~/.kube/config

- `experiments` contains the configuration for federated learning experiments.

    `default` contains the default settings for all experiments in the list.

key          | description | mandatory|default_value
------------ | ----------- |----------|------------
`exec_mode`  | run experiments in sequential or parallel mode, options: `[seq,parallel]`  | no | seq
`image_name`| Name of IBMFL image installed in OpenShift image streams  | no | ibmfl:latest
`commands:aggregator`  | FL commands to execute from aggregator pod | no | ['START', 'TRAIN', 'EVAL', 'STOP']

Each item in the experiment list consist of following keys

key          | description | mandatory|default_value
------------ | ----------- |----------|------------
`staging_dir`  | local directory where experiment config and data files are stored  | yes |
`name`| unique name for experiment  | no | ibmfl
`num_trials`  | number of trials for experiment | no | 1
`cluster_list`  | List of OpenShift clusters connection details | yes |
`data:pvc_name`  | persistent volume claim name for COS bucket which holds the datasets and model artifacts | no |

For COS bucket access, please set up Persistent Volume (PV) and Persistent Volume Claim (PVC) in OpenShift Cluster and provide the PVC name in `data:pvc_name` key.

Each item in the cluster list consist of following keys

key          | description | mandatory|default_value
------------ | ----------- |----------|------------
`context_name`  | context name of OpenShift Cluster as defined in kube config file  | yes |
`namespace`| namespace of OpenShift Cluster as defined in kube config file  | yes |

If multiple openshift clusters are configured, the aggregator pod will be deployed in the first cluster of the list and parties will be equally split among clusters.

### Debug

1. Orchestrator logs can be found at `error.log` and `info.log` for debugging.
2. Orchestrator collects aggregator and party logs inside experiment folder to debug aggregator and party training processes.
3. In the event orchestrator process gets stuck at pod creation step, you can use the follow commands to debug

- `oc get pods` to view pod status
- `oc logs -f [pod_name]` to view pod logs
- `oc delete pod [pod_name]` to delete any hanging pods
