"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2023 All Rights Reserved.
"""
import logging
import os
import pathlib

import yaml
from kubernetes import client, config, watch
from kubernetes.client import ApiException
from kubernetes.stream import stream
from openshift.dynamic import DynamicClient

logger = logging.getLogger(__name__)
import subprocess
import tarfile
from tempfile import TemporaryFile


class FLSpawner:
    """
    FLSpawner creates and manage the FL aggregator and party pods in a kubernetes cluster \
    using the kubernetes client apis.
    """

    def __init__(self, cluster, namespace, config_file=None, context=None, data=None):
        """
        Instantiate the FLSpawner based on the cluster info and kube config file , \
        kube config file will be generated by kubernetes client when you setup the \
        credentials to access the cluster
        :param cluster: cluster dictionary contains information about namespace, \
        cpu and memory requirements
        :param namespace: namespace of the cluster
        :param config_file: user provided config file, if not provided \
        use the default kube config file
        :param context: context of the cluster
        :param data: pvc(persistent volume claim) name for cos bucket
        """

        self.k8s_client = config.new_client_from_config(config_file, context, True)
        self.dynamic_client = DynamicClient(self.k8s_client)
        self.cluster = cluster
        self.namespace = namespace
        self.config_file = config_file
        self.context = context
        self.data = data
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        if data is None:
            with open(os.path.join(__location__, "pod_template.yml")) as pod_tmpl_file:
                self.pod_tmpl = pod_tmpl_file.read()
        else:
            with open(os.path.join(__location__, "pod_persistence_agg_template.yml")) as pod_tmpl_file:
                self.pod_agg_tmpl = pod_tmpl_file.read()
            with open(os.path.join(__location__, "pod_persistence_party_template.yml")) as pod_tmpl_file:
                self.pod_party_tmpl = pod_tmpl_file.read()
        with open(os.path.join(__location__, "service_template.yml")) as service_tmpl_file:
            self.service_tmpl = service_tmpl_file.read()
        with open(os.path.join(__location__, "route_template.yml")) as route_tmpl_file:
            self.route_tmpl = route_tmpl_file.read()

    def create_pod(
        self, pod_name, image_name, role, command_list, cos_mount_path, cpu="2", memory="4Gi", aggregator=False
    ):
        """
        Create pod in a kubernetes cluster based on the pod_template file
        :param pod_name: string to specify the name of the pod
        :param image_name: name of the docker image to create the pod
        :param role: role to group the pods
        :param command_list: pod start commands
        :param cpu: cpu required to run the pod
        :param memory: memory required to run the pod
        :param cos_mount_path: data mount path in the pod to mount the pvc
        """
        if aggregator:
            if self.data is None:
                pod = self.pod_tmpl.format(pod_name, pod_name, self.namespace, image_name, cpu, memory, command_list)
            else:
                pod = self.pod_agg_tmpl.format(
                    pod_name,
                    pod_name,
                    self.namespace,
                    image_name,
                    cos_mount_path,
                    cpu,
                    memory,
                    command_list,
                    self.data.get("pvc_name"),
                )
        else:
            if self.data is None:
                pod = self.pod_tmpl.format(pod_name, pod_name, self.namespace, image_name, cpu, memory, command_list)
            else:
                pod = self.pod_party_tmpl.format(
                    pod_name,
                    pod_name,
                    self.namespace,
                    image_name,
                    cos_mount_path,
                    cpu,
                    memory,
                    command_list,
                    self.data.get("pvc_name"),
                )
        v1_pod = self.dynamic_client.resources.get(api_version="v1", kind="Pod")
        pod_data = yaml.safe_load(pod)

        logger.info(f"create_pod {pod_data}")

        resp = v1_pod.create(body=pod_data, namespace=self.namespace)

    def get_pod_ip(self, pod_name):
        """
        Get pod ip address by pod name
        :param pod_name: name of pod to fetch the ip address
        :return: pod ip
        """
        v1_pods = self.dynamic_client.resources.get(api_version="v1", kind="Pod")

        res = v1_pods.get(name=pod_name, namespace=self.namespace)
        return res.status.podIP

    def get_route_url(self, pod_name):
        """
        Get pod ip address by pod name
        :param pod_name: name of pod to fetch the ip address
        :return: route URL
        """
        v1_route = self.dynamic_client.resources.get(api_version="route.openshift.io/v1", kind="Route")

        res = v1_route.get(name=pod_name, namespace=self.namespace)
        return "https://{}".format(res.spec.host)

    def get_pod_status(self, pod_name):
        """
        Get pod status by pod name
        :param pod_name:
        :return: return the pod states, pod states can be [Waiting, Running , Terminated]
        """
        v1_pods = self.dynamic_client.resources.get(api_version="v1", kind="Pod")
        res = v1_pods.get(name=pod_name, namespace=self.namespace)
        return res.status.phase

    def create_service(self, pod_name):
        """
        Create service for pods
        :param pod_name: pod name
        :return: status of service
        """
        service = self.service_tmpl.format(pod_name, pod_name)
        v1_services = self.dynamic_client.resources.get(api_version="v1", kind="Service")
        service_data = yaml.safe_load(service)
        resp = v1_services.create(body=service_data, namespace=self.namespace)

    def create_route(self, pod_name):
        """
        Create routes for the pods
        :param pod_name: name of the pod
        :return: status of the routes
        """
        route = self.route_tmpl.format(pod_name, pod_name, pod_name)
        v1_routes = self.dynamic_client.resources.get(api_version="route.openshift.io/v1", kind="Route")
        route_data = yaml.safe_load(route)
        resp = v1_routes.create(body=route_data, namespace=self.namespace)

    def execute_copy_commands(self, name, source_file, destination_file):
        """
        NOTE: this method is currently not used since cannot handle \
        copy of large data files
        Copy files from local machine to pods, deprecated using there is \
        issue when copying training files
        :param name: name of the pod
        :param source_file: source file
        :param destination_file: destination file
        """
        core_v1 = client.CoreV1Api(self.k8s_client)
        try:
            exec_command = ["tar", "xvf", "-", "-C", "/"]
            api_response = stream(
                core_v1.connect_get_namespaced_pod_exec,
                name,
                self.namespace,
                command=exec_command,
                stderr=True,
                stdin=True,
                stdout=True,
                tty=False,
                _preload_content=False,
            )

            with TemporaryFile() as tar_buffer:
                with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                    tar.add(source_file, destination_file)

                tar_buffer.seek(0)
                commands = []
                commands.append(tar_buffer.read())

                while api_response.is_open():
                    api_response.update(timeout=1)
                    if api_response.peek_stdout():
                        print("STDOUT: %s" % api_response.read_stdout())
                    if api_response.peek_stderr():
                        print("STDERR: %s" % api_response.read_stderr())
                    if commands:
                        c = commands.pop(0)
                        api_response.write_stdin(c.decode())
                    else:
                        break
                api_response.close()
        except ApiException as e:
            print("Exception when copying file to the pod%s \n" % e)

    def execute_shell_commands(self, name, sh_cmd_list):
        """
        Execute the shell commands passed as parameter inside a running pod
        :param name: name of pod to run the shell command
        :param sh_cmd_list: sh command list
        """
        exec_command = ["/bin/sh"]
        # copy from local to remote
        core_v1 = client.CoreV1Api(self.k8s_client)
        resp = stream(
            core_v1.connect_get_namespaced_pod_exec,
            name,
            self.namespace,
            command=exec_command,
            stderr=True,
            stdin=True,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        commands = sh_cmd_list

        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                logger.info("STDOUT: %s" % resp.read_stdout())
            if resp.peek_stderr():
                logger.info("STDERR: %s" % resp.read_stderr())
            if commands:
                c = commands.pop(0)
                resp.write_stdin(c + "\n")
            else:
                break

        resp.close()

    def copy_files(self, pod_name, src_file, dest_file):
        """
        Copy files from local machine to running pod using kubectl cp command
        :param pod_name: name of the pod where the files to be copied
        :param src_file: absolute path of source file
        :param dest_file: absolute path of destination file
        """
        if self.config_file is None:
            os.system("kubectl cp {} {}/{}:{}".format(src_file, self.namespace, pod_name, dest_file))
        else:
            os.system(
                "kubectl config --kubeconfig={} use-context {} && kubectl --kubeconfig={} cp  {} {}/{}:{}".format(
                    self.config_file, self.context, self.config_file, src_file, self.namespace, pod_name, dest_file
                )
            )

    def copy_files_from_pod(self, pod_name, remote_filepath, local_filepath):
        """
        Copy files from remote pod to the local staging directory using kubectl cp command
        :param remote_filepath: absolute path of the file in the pod to copy
        :param pod_name: name of the pod where the file resides
        :param local_dir: absolute path of local directory
        :return: None
        """
        if self.config_file is None:
            ls_output = "kubectl exec {} -- ls {}".format(pod_name, pathlib.Path(remote_filepath).parent)
        else:
            ls_output = (
                "kubectl config --kubeconfig={} use-context {} && kubectl --kubeconfig={} exec {} -- ls {}".format(
                    self.config_file, self.context, self.config_file, pod_name, pathlib.Path(remote_filepath).parent
                )
            )

        process_ls = subprocess.run(
            ls_output, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        if process_ls.returncode != 0:
            logger.error("Erred running ls on pod: ", process_ls.stderr)
            return False
        else:
            logger.info("Checking for existence of filepath:{} in Pod:{}".format(remote_filepath, pod_name))
            files_present = process_ls.stdout.split("\n")
            file_name = pathlib.Path(remote_filepath).name
            if file_name not in files_present:
                logger.info("Remote file:{} not found on Pod:{}".format(file_name, pod_name))
                return False

        if self.config_file is None:
            cmd = "kubectl cp {}/{}:{} {}".format(self.namespace, pod_name, remote_filepath, local_filepath)
        else:
            cmd = "kubectl config --kubeconfig={} use-context {} && kubectl --kubeconfig={} cp {}/{}:{} {}".format(
                self.config_file,
                self.context,
                self.config_file,
                self.namespace,
                pod_name,
                remote_filepath,
                local_filepath,
            )

        logger.debug("Copying from pod by running: {}".format(cmd))

        process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            logger.error("Erred: ", process.stderr)
            return False
        else:
            logger.info("Copied file: {} from: {} to: {}".format(remote_filepath, pod_name, local_filepath))
            return True

    def delete_pod(self, pod_name):
        """
        Delete a running pod by pod name
        :param pod_name: name of the pod to be deleted
        """
        try:
            core_v1 = client.CoreV1Api(self.k8s_client)
            core_v1.delete_namespaced_pod(name=pod_name, namespace=self.namespace, body=client.V1DeleteOptions())
        except ApiException as e:
            print("Error deleting pod {}".format(e))

    def delete_service(self, service_name):
        """
        Delete a running pod by pod name
        :param service_name: name of the service to be deleted
        """
        try:
            core_v1 = client.CoreV1Api(self.k8s_client)
            core_v1.delete_namespaced_service(name=service_name, namespace=self.namespace)
        except Exception as e:
            logger.debug("Error deleting service {}".format(e))

    def delete_routes(self, route_name):
        """
        Delete routes by route name
        :param route_name: name of the route to be deleted
        """
        try:
            dynamic_client = DynamicClient(self.k8s_client)
            v1_services = dynamic_client.resources.get(api_version="route.openshift.io/v1", kind="Route")
            v1_services.delete(name=route_name, namespace=self.namespace)
        except Exception as e:
            logger.debug("Error deleting route {}".format(e))

    def get_logs_from_pod(self, pod_name, log_path):
        """
        Configure kubernetes watcher for the pod so that the pod logs \
        will be streamed to the log file
        :param pod_name: name of the pod to get the logs
        :param log_path: path of the log file
        """
        with open(log_path, "a+") as config_file:
            try:
                core_v1 = client.CoreV1Api(self.k8s_client)
                w = watch.Watch()
                for e in w.stream(
                    core_v1.read_namespaced_pod_log,
                    name=pod_name,
                    namespace=self.namespace,
                    follow=True,
                    _preload_content=False,
                ):
                    config_file.write("{}\n".format(e))
            except Exception as e:
                print(e)
            finally:
                w.stop()

    def watch_pods(self, pod_name, log_path):
        v1_pod = self.dynamic_client.resources.get(api_version="v1", kind="Pod")

        # Prints the resource that triggered each event related to Services in the 'test' namespace
        with open(log_path, "a+") as config_file:
            for event in v1_pod.watch(namespace=self.namespace, name=pod_name):
                config_file.write("{}\n".format(event))

    def spawn_aggregator(self, pod_name, pod_staging_dir, cos_mount_path, image_name):
        """
        Spawn a FL aggregator as a pod and start the aggregator
        :param pod_name: name of the aggregator pod
        :param pod_staging_dir: pod staging directory to load the configs and datasets for training
        :param image_name: FL docker image name to create the aggregator pod
        """
        cpu = self.cluster["agg_pod"]["cpu"] or "2"
        memory = self.cluster["agg_pod"]["memory"] or "4Gi"
        image_name = image_name or "ibmfl:latest"
        label_role = "ibmfl"
        command_list = ["python3", "/FL/openshift_fl/run_agg.py", "{}/config_agg.yml".format(pod_staging_dir)]
        self.create_pod(pod_name, image_name, label_role, command_list, cos_mount_path, cpu, memory, aggregator=True)

    def spawn_party(self, pod_name, party_index, pod_staging_dir, cos_mount_path, image_name):
        """
        Spawn a FL party as a pod and start the party
        :param pod_name: name of the party pod
        :param party_index: index to identify the party when run multiple parties
        :param pod_staging_dir: pod staging directory to load the configs and datasets for training
        :param image_name: FL docker image name to create party pod
        """
        cpu = self.cluster["party_pod"]["cpu"] or "2"
        memory = self.cluster["party_pod"]["memory"] or "4Gi"
        image_name = image_name or "ibmfl:latest"
        label_role = "ibmfl"
        command_list = [
            "python3",
            "/FL/openshift_fl/run_party.py",
            "{}/config_party{}.yml".format(pod_staging_dir, party_index),
        ]
        self.create_pod(pod_name, image_name, label_role, command_list, cos_mount_path, cpu, memory, aggregator=False)

    def copy_dataset_configs_to_pods(self, pod_name, file_list, pod_staging_dir, commands=None):
        """
        copy datasets and config files to pods to run the experiments
        :param pod_name: name of the pod to copy the files
        :param file_list: source file list to be copied to the pod
        :param pod_staging_dir: pod staging directory to copy the files and datasets
        :param commands: commands configured by user to run as part of training, commands \
               can be  ['START', 'TRAIN', 'SAVE','EVAL','STOP']
        """
        end_of_file_marker = ["echo copied >> /tmp/end_of_file_marker.txt"]

        create_trial_dir_command = ["mkdir {}".format(pod_staging_dir)]
        self.execute_shell_commands(pod_name, create_trial_dir_command[:])
        logger.info("Creating trial directory in pod - {} completed".format(pod_name))
        for file in file_list:
            file_base_name = os.path.basename(file)
            if self.data is None:
                self.copy_files(pod_name, file, "{}/{}".format(pod_staging_dir, file_base_name))
                logger.info("Copying file {} from local to  pod - {} completed".format(file_base_name, pod_name))
            else:
                if file_base_name.endswith(".yml"):
                    self.copy_files(pod_name, file, "{}/{}".format(pod_staging_dir, file_base_name))
                    logger.info("Copying yml file {} from local to pod - {} completed".format(file_base_name, pod_name))

        if commands is not None:
            commands_str = ["echo {} >> /tmp/commands.txt".format(commands)]
            self.execute_shell_commands(pod_name, commands_str[:])
        self.execute_shell_commands(pod_name, end_of_file_marker[:])
