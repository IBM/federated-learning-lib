"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2023 All Rights Reserved.
"""
import asyncio
import concurrent.futures
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import numpy as np
import yaml
from data_copy_util import stage_trial_files
from kubernetes import client, watch
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    ExperimentRunner runs the federated training experiments \
    - Cordinates execution of FL commands between aggregator and party pods \
    - Executes multiple trials of experiment \
    - Captures the experiment trace from aggregator and party pods \
    """

    def __init__(self, default, experiment, fl_spawner_dict):
        self.default = default
        self.experiment = experiment
        self.fl_spawner_dict = fl_spawner_dict
        self.aggregator_commands = None
        self.image_name = self.get_image_name(default, experiment)
        self.aggregator_commands = self.get_aggregator_commands(default)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        name = self.experiment.get("name") or "ibmfl"
        self.exp_id = "{}-{}".format(name, ts)

    def get_image_name(self, default, experiment):
        """
        Get image name to spawn the FL aggregator and party pods from config
        :param default: dictionary contains the default configuration for all the experiments
        :param experiment: dictionary contains info to run the experiment
        :return:
        """
        image_name = "ibmfl:latest"
        if default is not None:
            if default.get("image_name") is not None:
                image_name = default.get("image_name")
        image_name = experiment.get("image_name") or image_name
        return image_name

    def get_aggregator_commands(self, default):
        """
        Get aggregator commands from config
        :param default: default configuration for all experiments
        :return: aggregator commands eg ['START', 'TRAIN', 'SAVE','EVAL','STOP']
        """
        aggregator_commands = None
        if default is not None:
            commands = default.get("commands") or None
            if commands is not None:
                aggregator_commands = commands.get("aggregator") or None
        return aggregator_commands

    def get_aggregator_pod_name(self, trial_id):
        """
        Returns aggregator pod name, its a combination of experiment id, 'agg' string and trial id
        :param trial_id: integer represent trial id
        :return: string
        """
        return "{}-agg-{}".format(self.exp_id, trial_id)

    def get_party_pod_name(self, party_index, trial_id):
        """
        Returns party pod name, its a combination of experiment_id, 'party' string, party index and trial id
        :param party_index: integer represent party number
        :param trial_id: integer represent trial id
        :return: string
        """
        return "{}-party{}-{}".format(self.exp_id, party_index, trial_id)

    def run_experiment(self):
        """
        Launches the experiment and runs multiple trials.
        """
        logger.info("Launching experiment with ID {}".format(self.exp_id))
        trials_tot = self.experiment.get("num_trials") or 1
        for trial in range(trials_tot):
            self.run_trial(trial)
        logger.info("Experiment {} completed ...............................................".format(self.exp_id))

    def run_trial(self, trial_id):
        """
        Run the trial of the experiment
        :param trial_id: integer represent trial id
        """
        logger.info("Trial {} run of {} started".format(trial_id, self.exp_id))
        local_staging_dir = self.experiment["staging_dir"]
        exp_root_dir = "{}/{}".format(local_staging_dir, self.exp_id)
        exp_trial_dir = "{}/{}".format(exp_root_dir, "trial_{}".format(trial_id))
        exp_log_dir = "{}/{}".format(exp_trial_dir, "logs")
        Path(exp_trial_dir).mkdir(parents=True, exist_ok=True)
        os.mkdir(exp_log_dir)
        exp_artifacts_dir = "{}/{}".format(exp_trial_dir, "staging")
        pod_staging_dir = "/FL/staging_dir"
        cos_mount_path = "/FL/datasets"

        # TODO exp_artifacts_dir is used to refer the agg file, not a clean design
        if self.experiment.get("data") is None:
            proc_file_map = stage_trial_files(local_staging_dir, exp_artifacts_dir, pod_staging_dir)
        else:
            proc_file_map = stage_trial_files(local_staging_dir, exp_artifacts_dir, cos_mount_path)

        agg_pod = self.get_aggregator_pod_name(trial_id)
        # create agg pod
        agg_cluster_context = list(self.fl_spawner_dict.keys())[0]
        config_agg_dict = self.init_aggregator(
            agg_cluster_context,
            agg_pod,
            self.aggregator_commands,
            pod_staging_dir,
            exp_artifacts_dir,
            cos_mount_path,
            self.image_name,
            proc_file_map,
            exp_log_dir,
        )

        # create agg service
        # todo agg port from config

        self.fl_spawner_dict[agg_cluster_context].create_service(agg_pod)

        logger.info("Aggregator service created")

        # create agg route
        self.fl_spawner_dict[agg_cluster_context].create_route(agg_pod)
        logger.info("Aggregator route created")

        # get agg url
        # todo rename variable
        agg_ip = self.fl_spawner_dict[agg_cluster_context].get_route_url(agg_pod)

        # create party pods
        n_parties = config_agg_dict.get("hyperparams").get("global").get("num_parties")

        # Distributed parties to cluster

        fl_party_scheduler = self.schedule_parties_cluster(n_parties, self.fl_spawner_dict)

        aggregator_started = False
        pods_started = False
        pods_stopped = False
        w = None
        try:
            core_v1 = client.CoreV1Api(self.fl_spawner_dict[agg_cluster_context].k8s_client)
            w = watch.Watch()
            for agg_outline in w.stream(
                core_v1.read_namespaced_pod_log,
                name=agg_pod,
                namespace=self.fl_spawner_dict[agg_cluster_context].namespace,
            ):
                if "Aggregator start successful" in agg_outline:
                    aggregator_started = True
                    logger.info("Agg pod - {}-agg running successfully".format(agg_pod))
                if aggregator_started and not pods_started:
                    pods_started = self.init_parties(
                        trial_id,
                        agg_ip,
                        pod_staging_dir,
                        exp_artifacts_dir,
                        cos_mount_path,
                        self.image_name,
                        proc_file_map,
                        exp_log_dir,
                        fl_party_scheduler,
                    )
                if "Aggregator stop successful" in agg_outline:
                    logger.info("Aggregator stop successful")
                    for party_index in fl_party_scheduler:
                        party_pod_name = self.get_party_pod_name(party_index, trial_id)
                        fl_party_scheduler[party_index].delete_pod(party_pod_name)
                        logger.info("Party {} pod deleted".format(party_pod_name))
                        fl_party_scheduler[party_index].delete_service(party_pod_name)
                        logger.info("Party {} service deleted".format(party_pod_name))
                        fl_party_scheduler[party_index].delete_routes(party_pod_name)
                        logger.info("Party {} route deleted".format(party_pod_name))

                    self.fl_spawner_dict[agg_cluster_context].delete_pod(agg_pod)
                    logger.info("Aggregator pod deleted")
                    self.fl_spawner_dict[agg_cluster_context].delete_service(agg_pod)
                    logger.info("Aggregator service deleted")
                    self.fl_spawner_dict[agg_cluster_context].delete_routes(agg_pod)
                    logger.info("Aggregator route deleted")
                    pods_stopped = True

                    break
        except ApiException as e:
            logger.error(e)
            if not pods_stopped:
                logger.error(e)

        except ValueError as e:
            logger.error(e)

        except Exception as e:
            logger.error(e)

        finally:
            w.stop()

        logger.info("Trial {} run of {} completed ..................................".format(trial_id, self.exp_id))

    def schedule_parties_cluster(self, n_parties, fl_spawner_dict):
        """
        In a multi cluster mode, the parties are split \
        equally among clusters
        :param n_parties: number of parties to split
        :param fl_spawner_dict: dictionary of fl spawner
        :return: return party scheduler with party to fl_spawner mapping
        """
        fl_party_scheduler = {}
        cluster_size = len(self.fl_spawner_dict)
        party_split = np.array_split(range(n_parties), cluster_size)
        index = -1
        for key in self.fl_spawner_dict:
            index += 1
            for party_index in party_split[index]:
                fl_party_scheduler[party_index] = fl_spawner_dict[key]
        return fl_party_scheduler

    def init_parties(
        self,
        trial_id,
        agg_ip,
        pod_staging_dir,
        exp_artifacts_dir,
        cos_mount_path,
        ibmfl_image,
        proc_file_map,
        exp_log_dir,
        fl_party_scheduler,
    ):
        """
        Initialize the FL parties configured in the experiment, all parties are created and managed concurrently.
        :param trial_id: integer represent the trial id
        :param agg_ip: ip address of the aggregator
        :param pod_staging_dir: staging directory of the pod to copy the configs and datasets
        :param exp_artifacts_dir: local path where the config files and datasets are stored
        :param cos_mount_path: data mount path in pods
        :param ibmfl_image: docker image name to create the pod
        :param n_parties: number of parties need to be created
        :param proc_file_map: map contains the config files and datasets path
        :param exp_log_dir: log directory path to copy the pod logs
        :return: status of the party pod created
        """
        # TODO : Spawn parties in parallel by using multi threading
        for key in fl_party_scheduler:
            self.init_party(
                trial_id=trial_id,
                agg_ip=agg_ip,
                pod_staging_dir=pod_staging_dir,
                exp_artifacts_dir=exp_artifacts_dir,
                cos_mount_path=cos_mount_path,
                ibmfl_image=ibmfl_image,
                party_index=key,
                proc_file_map=proc_file_map,
                exp_log_dir=exp_log_dir,
                fl_spawner=fl_party_scheduler[key],
            )

        return True

    def init_party(
        self,
        trial_id,
        agg_ip,
        pod_staging_dir,
        exp_artifacts_dir,
        cos_mount_path,
        ibmfl_image,
        party_index,
        proc_file_map,
        exp_log_dir,
        fl_spawner,
    ):
        """
        creates the FL party pods , copy config and datasets to the pod and \
        configure to stream the pod logs
        to the local log directory
        :param trial_id: integer represent the trial id
        :param agg_ip: ip address of the aggregator
        :param pod_staging_dir: staging directory of the pod to copy the configs and datasets
        :param exp_artifacts_dir: local path of the  config files and datasets are stored
        :param cos_mount_path: data mount path in pods
        :param ibmfl_image: docker image name to create the pod
        :param party_index: integer to identify the party number
        :param proc_file_map: map contains the config files and datasets path
        :param exp_log_dir: log directory path to copy the pod logs
        """
        party_pod_name = self.get_party_pod_name(party_index, trial_id)
        fl_spawner.spawn_party(party_pod_name, party_index, pod_staging_dir, cos_mount_path, ibmfl_image)
        logger.info("Party pod {} created successfully".format(party_pod_name))
        party_status = fl_spawner.get_pod_status(party_pod_name)
        while party_status != "Running":
            sleep(10)
            party_status = fl_spawner.get_pod_status(party_pod_name)
        logger.info("Party pod {} running successfully".format(party_pod_name))
        # create service
        fl_spawner.create_service(party_pod_name)

        # create route
        fl_spawner.create_route(party_pod_name)
        party_ip = fl_spawner.get_route_url(party_pod_name)
        logger.info("Party {} route - {}".format(party_pod_name, party_ip))
        config_party_path = "{}/config_party{}.yml".format(exp_artifacts_dir, party_index)
        with open(config_party_path) as config_party_file:
            config_party_str = config_party_file.read()
        config_party_dict = yaml.load(config_party_str, Loader=yaml.Loader)
        del config_party_dict["aggregator"]["ip"]
        del config_party_dict["aggregator"]["port"]
        config_party_dict["connection"]["info"]["ip"] = "0.0.0.0"
        config_party_dict["connection"]["info"]["port"] = 5000
        config_party_dict["aggregator"]["url"] = agg_ip
        config_party_dict["connection"]["info"]["url"] = party_ip
        with open(config_party_path, "w") as config_file:
            yaml.dump(config_party_dict, config_file)
        fl_spawner.copy_dataset_configs_to_pods(
            party_pod_name, proc_file_map["party{}".format(party_index)], pod_staging_dir
        )
        logger.info("Copy dataset and configs to party pod {} completed".format(party_pod_name))
        # call to capture logs
        party_log_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        party_log_executor.submit(
            fl_spawner.get_logs_from_pod, party_pod_name, "{}/party-{}.out".format(exp_log_dir, party_index)
        )

    def init_aggregator(
        self,
        agg_cluster_context,
        agg_pod,
        agg_commands,
        pod_staging_dir,
        exp_artifacts_dir,
        cos_mount_path,
        ibmfl_image,
        proc_file_map,
        exp_log_dir,
    ):
        """
        Initialize the FL aggregator includes aggregator pod creation, copy config files and datasets to \
        aggregator pod, run the experiment, configure streaming of the aggregator pod logs
        :param agg_pod: name of the aggregator pod to be created
        :param agg_commands: commands to run the training configured by the user
        :param pod_staging_dir: staging directory of the pod to copy the configs and datasets
        :param exp_artifacts_dir: local path of the  config files and datasets are stored
        :param ibmfl_image: docker image name to create the pod
        :param proc_file_map: map contains the config files and datasets path
        :param exp_log_dir: log directory path to copy the pod logs
        :param cos_mount_path: data mount path in pods
        :return: aggreagator config file
        """

        self.fl_spawner_dict[agg_cluster_context].spawn_aggregator(
            agg_pod, pod_staging_dir, cos_mount_path, ibmfl_image
        )
        logger.info("Agg pod - {}-agg created successfully".format(self.exp_id))

        # check agg status
        pod_status = self.fl_spawner_dict[agg_cluster_context].get_pod_status(agg_pod)
        while pod_status != "Running":
            sleep(10)
            pod_status = self.fl_spawner_dict[agg_cluster_context].get_pod_status(agg_pod)
        # update agg ip to 0.0.0.0
        config_agg_path = "{}/config_agg.yml".format(exp_artifacts_dir)
        with open(config_agg_path) as config_agg_file:
            config_agg_str = config_agg_file.read()
        config_agg_dict = yaml.load(config_agg_str, Loader=yaml.Loader)
        config_agg_dict["connection"]["info"]["ip"] = "0.0.0.0"
        with open(config_agg_path, "w") as config_file:
            yaml.dump(config_agg_dict, config_file)
        # copy staging dir
        self.fl_spawner_dict[agg_cluster_context].copy_dataset_configs_to_pods(
            agg_pod, proc_file_map["agg"], pod_staging_dir, agg_commands
        )
        logger.info("Copy datasets and configs completed for aggregator pod - {} completed".format(agg_pod))
        agg_log_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        agg_log_executor.submit(
            self.fl_spawner_dict[agg_cluster_context].get_logs_from_pod,
            pod_name=agg_pod,
            log_path="{}/agg.out".format(exp_log_dir),
        )

        return config_agg_dict
