"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""
import concurrent.futures
import logging
import os
import sys

import re
import yaml
from experiment_runner import ExperimentRunner
from fl_spawner import FLSpawner

fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)

from ibmfl.util.config import configure_logging_from_file

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator runs FL experiments in OpenShift clusters
    """

    def __init__(self, config_global):
        """
        Instantiate the orchestrator based on the config file provided \
        :param config_global: path to yml file that contains the cluster \
        and experiment info
        """
        configure_logging_from_file()
        self.cluster = config_global.get('cluster') or None
        self.experiments = config_global.get('experiments') or None
        self.validate_config()

    def start(self):
        """
        Launches the experiments configured in the yml file either in \
        sequential or parallel execution mode
        """
        exec_mode = None
        self.config_file = self.cluster.get('kube_config_location')
        experiment_default = self.experiments.get('default') or None
        if experiment_default is not None:
            exec_mode = experiment_default.get('exec_mode') or None
        experiment_list = self.experiments.get('experiment_list') or []
        if exec_mode is not None and exec_mode == 'parallel':
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                for experiment in experiment_list:
                    fl_spawner_dict = self.get_cluster_configs(experiment)
                    experiment_runner = ExperimentRunner(experiment_default, experiment, fl_spawner_dict)
                    executor.submit(experiment_runner.run_experiment)
        else:
            for experiment in experiment_list:
                fl_spawner_dict = self.get_cluster_configs(experiment)
                experiment_runner = ExperimentRunner(experiment_default, experiment, fl_spawner_dict)
                experiment_runner.run_experiment()

    def get_cluster_configs(self, experiment):
        """
        Retrieve list of cluster connect details from orchestrator config
        :param experiment: experiment info
        :return: dictionary (key - context_name of cluster, \
        value - cluster info)
        """
        data=experiment.get('data') or None
        cluster_list = experiment.get('cluster_list') or None
        fl_spawner_dict = {}
        for cluster in cluster_list:
            context_name = cluster.get('context_name')
            namespace = cluster.get('namespace')
            fl_spawner_dict[context_name] = FLSpawner(self.cluster, namespace, self.config_file, context_name, data)
        return fl_spawner_dict

    def validate_config(self):
        """
        Validate the orchestrator config file
        """
        if self.cluster is None:
            raise ValueError('Cluster cannot be empty')
        if self.experiments is None:
            raise ValueError('Experiments cannot be empty')
        if self.experiments.get('experiment_list') is None:
            raise ValueError('Experiment list cannot be empty')

        default = self.experiments.get('default')
        if default is not None:
            commands = default.get('commands')
            if commands is not None:
                agg_commands = commands.get('aggregator')
                if agg_commands is not None:
                    commands_list = ['START', 'TRAIN', 'EVAL', 'SAVE', 'STOP']
                    for agg_command in agg_commands:
                        if not any(agg_command.strip().upper() in s for s in commands_list):
                            raise ValueError('{} not a valid aggregator command'.format(agg_command))
                exec_mode = default.get('exec_mode')
                if exec_mode is not None:
                    exec_mode_list = ['seq', 'parallel']
                    if exec_mode not in exec_mode_list:
                        raise ValueError('{} not a valid exec mode'.format(exec_mode))

        regex = '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*'
        experiment_list = self.experiments.get('experiment_list')
        for experiment in experiment_list:
            exp_name = experiment.get('name')
            staging_dir = experiment.get('staging_dir')
            if exp_name is not None:
                if not re.match(regex, exp_name):
                    raise ValueError(
                        'Name must consist of lower case alphanumeric characters, '
                        '\'-\' or \'.\', and must start and end with an alphanumeric character')
            if staging_dir is None:
                raise ValueError('Staging dir cannot be empty for exp : ', exp_name)

            cluster_list = experiment.get('cluster_list') or None
            if cluster_list is None:
                raise ValueError('Atleast one cluster configuration required')
            else:
                for cluster in cluster_list:
                    context_name = cluster.get('context_name')
                    if context_name is None:
                        raise ValueError('Context name for cluster cannot be empty')
                    namespace = cluster.get('namespace')
                    if namespace is None:
                        raise ValueError('Namespace for cluster cannot be empty')
            data = experiment.get('data') or None
            if data is not None:
                pvc_name = data.get('pvc_name')
                if pvc_name is None:
                    raise ValueError('pvc_name for data cannot be empty')

if __name__ == '__main__':
    """
    Main function to create orchestrator instance \
    using yaml configuration file
    """
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        raise ValueError('Please provide yaml configuration')

    with open(sys.argv[1]) as config_global_file:
        config_global = yaml.load(config_global_file.read(), Loader=yaml.Loader)
    orchestrator = Orchestrator(config_global)
    orchestrator.start()
