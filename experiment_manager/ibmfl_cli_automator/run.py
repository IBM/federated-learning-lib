"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2023 All Rights Reserved.
"""
#!/usr/bin/env python3

import os
import pprint as pp
import random
import sys
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO, TextIOWrapper
from pathlib import Path

sys.path.append("../../")
import subprocess as sp
import time
from string import Template

import paramiko
import yaml
from tqdm.auto import tqdm

fl_path = os.path.abspath(".")
if fl_path not in sys.path:
    sys.path.append(fl_path)
import experiment_manager.ibmfl_cli_automator.postprocess as ibmfl_postproc

# USAGE:
# ./ibmfl_cli_automator/run_paramiko.py <runner_config_dir>
#
# ASSMUMPTIONS:
# - automator config is named config_runner.yml
# - config file templates are named config_agg_tmpl.yml, config_party_tmpl.yml
# TODO:
# X script should have args for the _dir variables, username of ssh, etc (in runner.yaml?)
# X fix agg_cmds and party_cmds being hard-coded as input args (obviously bad design)
# X separate paths for each experiment/trial?
# X scp output back to local machine so it's all gathered in same place
# X think about local config write location in copy_config_to_server


class Runner:
    """
    The runner class contains all the shared information about the ongoing runs, and has the capability of
    organizing the configuration of experiments where IBMFL runs are triggered in parameterized
    ways.
    """

    __cmds_agg = "START\nTRAIN\nEVAL\nSTOP"
    __cmds_party = "START\nREGISTER"

    def __init__(self):
        """
        Initialize various values that need to be shared as the experiments and trials take place
        """
        self.__agg_outstr = ""
        self.__started_training = False
        self.__trial_cur = 1
        self.__round_cur = 1
        self.__party_responses = []
        self.__exp_info = {}
        self.__config_party_dicts = []
        self.__exp_timestamp = None

    @staticmethod
    def generate_timestamp():
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    @staticmethod
    def __dump_trial_metadata(trial_info, config_agg_dict, config_party_dicts, ts, trial_cur):
        """
        Write a file with data used for monitoring the trials' processes

        :param trial_info: trial data
        :type trial_info: `dict`
        :param config_agg_dict: info for agg_machine as per the `config_runner.yml` file
        :type config_agg_dict: `list[dict]`
        :param config_party_dicts: info for party machines as per the `config_runner.yml` file
        :type config_party_dicts: `list[dict]`
        """
        trial_metadata = {}
        trial_metadata["timestamp"] = ts
        trial_metadata["local_staging_dir"] = trial_info["local_staging_dir"]
        trial_metadata["agg_machine"] = config_agg_dict["connection"]["info"]["ip"]
        trial_metadata["agg_username"] = trial_info["agg_machine"]["ssh_username"]
        trial_metadata["party_machines"] = []
        trial_metadata["party_usernames"] = []
        for pi, config_party_dict in enumerate(config_party_dicts):
            trial_metadata["party_machines"] += [config_party_dict["connection"]["info"]["ip"]]
            trial_metadata["party_usernames"] += [trial_info["party_machines"][pi]["ssh_username"]]
        with open("{}/{}/trial{}/metadata.yml".format(trial_info["local_staging_dir"], ts, trial_cur), "w") as mdfile:
            yaml.dump(trial_metadata, mdfile)

    @staticmethod
    def __exec_command_sync(client, cmdstr):
        """
        Run a command on the client and wait for it to finish, returning its return code and output

        :param client: remote client for executing the command
        :type client: `paramiko.client.SSHClient`
        :param cmdstr: entire command to run on client
        :type cmdstr: `str`
        :return: the status code as reported by paramiko, and the full stdout from the client
        :rtype: `tuple(int, str)`
        """
        # start command
        (stdin, stdout, stderr) = client.exec_command(cmdstr)

        # tell process we won't send anything to stdin
        stdin.channel.shutdown_write()
        outstr = ""
        errstr = ""
        # read output so we dont hang later on
        while not stdout.channel.exit_status_ready():
            while stdout.channel.recv_ready():
                outstr += stdout.channel.recv(1024).decode()
            while stderr.channel.recv_ready():
                errstr += stderr.channel.recv(1024).decode()
            time.sleep(1)
        # get exit status and rest of output
        status = stdout.channel.recv_exit_status()
        while stdout.channel.recv_ready():
            outstr += stdout.channel.recv(1024).decode()
        while stderr.channel.recv_ready():
            errstr += stdout.channel.recv(1024).decode()

        # get error message if exit status is bad
        if status != 0:
            ip, _ = client.get_transport().getpeername()
            sys.exit(
                f"Error executing command '{cmdstr}' on remote {ip} with exit code {status}!\nstdout: '{outstr}'\nstderr: '{errstr}'"
            )

        return (status, outstr)

    @staticmethod
    def __stat_on_server(client, path_remote):
        """
        Write file to server

        :param path_local: file on the local machine
        :type path_local: `str`
        :param client: remote client to copy the file to
        :type client: `paramiko.client.SSHClient`
        :param path_remote: place to copy the file to, on the remote machine
        :type path_remote: `str`
        :return: None
        """
        sftp = client.open_sftp()
        try:
            ret = sftp.stat(path_remote)
        except IOError:
            ret = 0
        sftp.close()
        return ret

    @staticmethod
    def copy_to_server_dir(src, dst, client):
        """
        Write file to server, in the specific case that its a directory.
        :param src: local path to be copied from
        :type src: `str`
        :param dst: destination path on the server to copy to
        :type dst: `str`
        :param client: file transfer client
        :type client: `paramiko.client.SSHClient`
        :return: None
        """
        sftp = client.open_sftp()
        for item in os.listdir(src):
            if os.path.isfile(os.path.join(src, item)):
                sftp.put(os.path.join(src, item), os.path.join(dst, item))
            else:
                # subdir within src
                try:
                    sftp.mkdir(os.path.join(dst, item))
                except IOError:
                    # print('Warn: IOError with mkdir at remote: Perhaps the directory already exists.')
                    # ignore if it exists
                    pass
                Runner.copy_to_server_dir(os.path.join(src, item), os.path.join(dst, item), client)

    @staticmethod
    def __copy_to_server(path_local, client, path_remote):
        """
        Write file to server, invokes copy_to_server_dir if path_local is a dir.
        :param path_local: file on the local machine
        :type path_local: `str`
        :param client: remote client to copy the file to
        :type client: `paramiko.client.SSHClient`
        :param path_remote: place to copy the file to, on the remote machine
        :type path_remote: `str`
        :return: None
        """
        # copy from local to remote
        if path_local == path_remote:
            return
        path_remote_dir = Path(path_remote).parent
        (status, outstr) = Runner.__exec_command_sync(client, "mkdir -p {}".format(path_remote_dir))
        sftp = client.open_sftp()
        print("copying {} to {}".format(path_local, path_remote))
        if os.path.isdir(path_local):
            if not os.path.exists(path_remote):
                try:
                    sftp.mkdir(path_remote)
                except IOError:
                    # suppress failure if directory exists
                    pass
            Runner.copy_to_server_dir(path_local, path_remote, client)
        else:
            sftp.put(path_local, path_remote)
        sftp.close()

    @staticmethod
    def __copy_from_server(path_remote, client, path_local):
        """
        Write file from server to local machine

        :param path_remote: file on the remote machine
        :type path_remote: `str`
        :param client: remote client to copy the file from
        :type client: `paramiko.client.SSHClient`
        :param path_local: place to copy the file to, on the local machine
        :type path_local: `str`
        :return: None
        """
        if path_remote == path_local:
            return
        config_path_dir = Path(path_local).parent
        config_path_dir.mkdir(parents=True, exist_ok=True)
        # copy from local to remote
        sftp = client.open_sftp()
        sftp.get(path_remote, path_local)
        sftp.close()

    @staticmethod
    def __start_ssh_connection(server_in, port_in, username_in):
        """
        Get paramiko handle for an SSH connection

        :param server_in: IP address or domain name to connect to
        :type server_in: string
        :param port_in: port to use for the connection
        :type port_in: string
        :param username_in: username to use for the connection
        :type username_in: string
        :return: object for interacting with that server
        :rtype: `paramiko.client.SSHClient`
        """
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.MissingHostKeyPolicy())
        client.connect(server_in, port=port_in, username=username_in)
        return client

    @staticmethod
    def __copy_config_to_server(config_dict, config_path_loc, client, config_path_rem):
        """
        Write config dict to local config path, then copy it to the remote

        :param config_dict: local config to send
        :type config_dict: `dict`
        :param config_path_loc: place to put the config on the local machine
        :type config_path_loc: `str`
        :param client: remote client to copy the config to
        :type client: `paramiko.client.SSHClient`
        :param config_path_rem: place to copy the config to, on the remote machine
        :type config_path_rem: `str`
        :return: None
        """
        # write local
        config_path_loc_dir = Path(config_path_loc).parent
        config_path_loc_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path_loc, "w") as config_file:
            yaml.dump(config_dict, config_file)
        # copy from local to remote
        if config_path_loc == config_path_rem:
            return
        config_path_rem_dir = Path(config_path_rem).parent
        (status, outstr) = Runner.__exec_command_sync(client, "mkdir -p {}".format(config_path_rem_dir))
        if status != 0:
            print("Bad config path.")
            exit(0)
        sftp = client.open_sftp()
        sftp.put(config_path_loc, config_path_rem)

        sftp.close()

    @staticmethod
    def __get_exec_string(machine_info, cmd, tag, ts, obtain_stdout):
        """
        Build the FL command we will run on the remote machine

        :param machine_info: info for machine as per the `config_runner.yml` file
        :type machine_info: `dict`
        :param cmd: FL command to execute, either `run_agg.py` or `run_party.py` for now
        :type cmd: `str`
        :param tag: tag for this IBMFL job (i.e. 'agg' or 'partyX' for now)
        :type tag: `str`
        :param ts: pseudo-unique generated timestamp for this trial
        :type ts: `str`
        :param obtain_stdout: whether we obtain stdout over the client, or just write it to file
        :type obtain_stdout: `boolean`
        :return: executable string for running the remote FL command appropriately on the given
        machine
        :rtype: `str`
        """
        # start job on server
        if machine_info["venv_uses_conda"]:
            python_str = "python"
            activate_str = "conda activate {} && ".format(machine_info["venv_dir"])
            deactivate_str = " && conda deactivate"
        else:
            python_str = "{}/bin/python".format(machine_info["venv_dir"])
            activate_str = ""
            deactivate_str = ""
        # build command
        command_str = "{python} {runner} {config}".format(
            python=python_str, runner=cmd, config="{}/{}/config_{}.yml".format(machine_info["staging_dir"], ts, tag)
        )

        redir = "| tee" if obtain_stdout else ">"

        # get timestamp and
        exec_string = "cd {dir} && {venvac}{command} 2> {stderr} {stdout}{venvde}".format(
            dir=machine_info["ibmfl_dir"],
            venvac=activate_str,
            command=command_str,
            stdout=f"{redir} {machine_info['staging_dir']}/{ts}/stdout_{tag}.txt",
            stderr=f"{machine_info['staging_dir']}/{ts}/stderr_{tag}.txt",
            venvde=deactivate_str,
        )
        return exec_string

    def __start_agg_job(self, config_agg_dict, agg_files, trial_info, ti, ts):
        """
        Start an aggregator job using the connection info supplied via the agg config

        :param config_agg_dict: info for agg_machine as per the `config_runner.yml` file
        :type config_agg_dict: `list[dict]`
        :param trial_info: trial data
        :type trial_info: `dict`
        :param ti: current trial (1 for first trial, 2 for second, etc.)
        :type ti: `int`
        :param ts: timestamp string for the trial
        :type ts: `str`
        """
        # start agg connection
        agg_ip = config_agg_dict["connection"]["info"]["ip"]
        agg_client = Runner.__start_ssh_connection(agg_ip, 22, trial_info["agg_machine"]["ssh_username"])
        print(f"Agg connection made to {agg_ip}")

        local_staging_dir = "{}/{}/trial{}".format(trial_info["local_staging_dir"], self.__exp_timestamp, ti)
        machine_staging_dir = "{}/{}".format(trial_info["agg_machine"]["staging_dir"], ts)

        # make remote staging dir
        status, outstr = Runner.__exec_command_sync(agg_client, "mkdir -p {}".format(machine_staging_dir))

        # copy config to server
        Runner.__copy_config_to_server(
            config_agg_dict, f"{local_staging_dir}/config_agg.yml", agg_client, f"{machine_staging_dir}/config_agg.yml"
        )
        for supp_file in agg_files:
            Runner.__copy_to_server(str(supp_file), agg_client, f"{machine_staging_dir}/{supp_file.name}")

        # start job on server
        agg_exec_string = Runner.__get_exec_string(
            trial_info["agg_machine"],
            "experiment_manager/ibmfl_cli_automator/run_agg.py",
            "agg",
            ts,
            obtain_stdout=True,
        )
        agg_handles = agg_client.exec_command(agg_exec_string)
        # print(f'Agg command {agg_exec_string} started')

        # write the commands to the job's stdin
        with BytesIO(Runner.__cmds_agg.encode("utf8")) as cmds_agg_file:
            with TextIOWrapper(cmds_agg_file, encoding="utf8") as cmds:
                for cmd in cmds:
                    agg_handles[0].write(cmd)
        agg_handles[0].channel.shutdown_write()
        return (agg_client, agg_handles)

    def __start_party_job(self, config_party_dict, party_files, trial_info, ti, pi, ts):
        """
        Start an party job using the connection info supplied via the party config

        :param config_agg_dict: info for agg_machine as per the `config_runner.yml` file
        :type config_agg_dict: `list[dict]`
        :param trial_info: trial data
        :type trial_info: `dict`
        :param pi: this party's numerical ID for the trial
        :type pi: `int`
        :param ts: timestamp string for the trial
        :type ts: `str`
        """
        # party i's connection
        party_ip = config_party_dict["connection"]["info"]["ip"]
        party_client = Runner.__start_ssh_connection(party_ip, 22, trial_info["party_machines"][pi]["ssh_username"])
        print("Party {} connection made to {}".format(pi, party_ip))

        # copy config to server
        local_staging_dir = "{}/{}/trial{}".format(trial_info["local_staging_dir"], self.__exp_timestamp, ti)
        machine_staging_dir = "{}/{}".format(trial_info["party_machines"][pi]["staging_dir"], ts)
        status, outstr = Runner.__exec_command_sync(party_client, "mkdir -p {}".format(machine_staging_dir))

        Runner.__copy_config_to_server(
            config_party_dict,
            f"{local_staging_dir}/config_party{pi}.yml",
            party_client,
            f"{machine_staging_dir}/config_party{pi}.yml",
        )

        for supp_file in party_files:
            Runner.__copy_to_server(str(supp_file), party_client, f"{machine_staging_dir}/{supp_file.name}")

        # start job on server
        party_exec_string = Runner.__get_exec_string(
            trial_info["party_machines"][pi],
            "experiment_manager/ibmfl_cli_automator/run_party.py",
            "party{}".format(pi),
            ts,
            obtain_stdout=False,
        )
        party_handles = party_client.exec_command(party_exec_string)
        # print(f'Party command {party_exec_string} started')

        # write the commands to the job's stdin
        with BytesIO(Runner.__cmds_party.encode("utf8")) as cmds_party_file:
            with TextIOWrapper(cmds_party_file, encoding="utf8") as cmds:
                for cmd in cmds:
                    party_handles[0].write(cmd)
        party_handles[0].channel.shutdown_write()
        return (party_client, party_handles)

    @staticmethod
    def __get_party_from_log(log, config_party_dicts):
        """
        Determine if one of the partys' respective IP addresses is in a portion of a log

        :param log: portion of a log obtained for an FL command's process via the paramiko client
        :type log: `str`
        :param config_party_dicts: info for party machines as per the `config_runner.yml` file
        :type config_party_dicts: `list[dict]`
        :return: index of party whose IP was found in the log, or -1 if none were found
        :rtype: `int`
        """
        for i, d in enumerate(config_party_dicts):
            if d["connection"]["info"]["ip"] in log:
                return i
        return -1

    def __display_status(
        self, exit_statuses, agg_stdout, config_party_dicts, round_progress, party_progress, n_parties
    ):
        """
        CLI output for the a trial

        :param exit_statuses: integers indicating whether agg and party jobs have exited
        :type exit_statuses: `list[int]`
        :param agg_stdout: paramiko stdout handle for the aggregator process
        :type agg_stdout: `paramiko.channel.Channel`
        :param config_party_dicts: info for party machines as per the `config_runner.yml` file
        :type config_party_dicts: `list[dict]`
        """
        # make sure we don't block and give some updates from agg
        if agg_stdout.recv_ready():
            self.__agg_outstr += agg_stdout.recv(1024).decode()
            while "\n" in self.__agg_outstr:
                agg_outline = self.__agg_outstr[: self.__agg_outstr.find("\n") + 1]
                party_ref = Runner.__get_party_from_log(agg_outline, config_party_dicts)
                if "Initiating Global Training" in agg_outline:
                    self.__started_training = True
                if self.__started_training:
                    if party_ref >= 0:
                        party_progress.update(1)
                        self.__party_responses += [Runner.__get_party_from_log(agg_outline, config_party_dicts)]
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if "Model update<" in agg_outline:
                        round_progress.update(1)
                        party_progress.reset(total=n_parties)
                        self.__round_cur += 1
                        self.__party_responses = []
                    if "Finished Global Training" in agg_outline:
                        round_progress.update(1)
                        round_progress.close()
                        party_progress.close()
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        sys.stdout.write("Done training at {}!\n".format(now))
                        sys.stdout.flush()
                self.__agg_outstr = self.__agg_outstr[self.__agg_outstr.find("\n") + 1 :]

        if any(exit_statuses):
            sys.stdout.write(
                "\rAgg finished: {} & Parties finished: {}".format(
                    "yes" if exit_statuses[-1] else "no", [vi for vi, v in enumerate(exit_statuses[:-1]) if v]
                )
            )
            sys.stdout.flush()

    @staticmethod
    def __perm_generator(seq_in, permlen):
        """
        Generator for permutations of elements, without giving the same permutation twice

        :param seq_in: elements to permute
        :type seq_in: `list`
        :param permlen: length of permutation to generate; can be shorter or longer than `len(seq_in)`
        :type permlen: `int`
        :return: the list of permuted elements, different than any previous call
        :rtype: `list`
        """
        seen = set()
        # TODO: don't enter infinite loop if all permutations have been seen
        while True:
            # make sure we have at least permlen elements, randomly chosen but minimal dups
            seq = []
            seq += seq_in
            while len(seq) < permlen:
                seq += random.sample(seq_in, permlen - len(seq))
            # permute the long-enough seq
            perm = tuple(random.sample(seq, permlen))
            # only take it if it's unique
            if perm not in seen:
                seen.add(perm)
                yield perm

    def get_machine_info(self, machine_label, machines):
        """
        Get all config from `config_runner.yml` necessary to run job on a machine, falling back to
        defaults

        :param machine_label: key in `config_runner.yml` corresponding to the machine of interest
        :type machine_label: `str`
        :param machines: raw machine info from `config_runner.yml`
        :type machines: `dict`
        :return: the fully filled-in machine info
        :rtype: `dict`
        """
        machine_info = {
            "ssh_username": None,
            "ip_address": None,
            "port_number": None,
            "staging_dir": None,
            "ibmfl_dir": None,
            "venv_dir": None,
            "venv_uses_conda": None,
        }
        default_conf = machines["default"] if "default" in machines else {}
        machine_conf = machines[machine_label]
        for key in machine_info.keys():
            if key in machine_conf:
                machine_info[key] = machine_conf[key]
            else:
                machine_info[key] = default_conf[key]
        return machine_info

    def get_trial_info(self, exp_info, machines, trial_nr):
        """
        Update and replace the values in the exp_info dictionary to contain a complete specification
        for the given trial of that experiment. Ultimately we are solidifying the exact machines
        used for the aggregator and each party, and in the exp_info dict the agg and party machines
        are listed by label, but we provide the full machine data at these keys for the trial_info
        dictionary instead.

        :param exp_info: experiment data
        :type exp_info: `dict`
        :param machines: list of machines that are available for use during the IBMFL runs
        for this experiment; a shared machine list can be specified for multiple experiments,
        and machines can be shuffled for the trials of a given experiment
        :type machines: `dict`
        :param trial_nr: the trial we're currently on (numerical, starting at 1)
        :type ui_mode: `int`
        :return: trial_info dictionary fully specifying the current trial to be run
        :rtype: `dict`
        """
        n_trials = exp_info["n_trials"]
        n_parties = exp_info["n_parties"]
        trial_info = deepcopy(exp_info)
        # if requested, generate randomized machines for parties
        if exp_info["shuffle_party_machines"]:
            machine_perm_gen = Runner.__perm_generator(exp_info["party_machines"], exp_info["n_parties"])
            machines_used = set()
            # TODO: don't perform this check if it's impossible
            while len(machines_used) < len(exp_info["party_machines"]):
                exp_party_machines = tuple(next(machine_perm_gen) for _ in range(n_trials))
                machines_used = set([item for sublist in exp_party_machines for item in sublist])
        else:
            exp_party_machines_tmp = exp_info["party_machines"]
            exp_party_machines_one = []
            exp_party_machines_one += exp_party_machines_tmp
            while len(exp_party_machines_one) < n_parties:
                exp_party_machines_one += exp_party_machines_tmp
            exp_party_machines_one = exp_party_machines_one[:n_parties]
            exp_party_machines = tuple(tuple(exp_party_machines_one) for _ in range(n_trials))

        # fill machines for this trial into configs
        # > change agg config
        trial_info["agg_machine"] = self.get_machine_info(exp_info["agg_machine"], machines)

        # > change party configs
        trial_info["party_machines"] = []
        for pi in range(n_parties):
            trial_info["party_machines"] += [self.get_machine_info(exp_party_machines[trial_nr][pi], machines)]

        return trial_info

    @staticmethod
    def __copy_logs_to_local(machine_staging_dir, client, local_staging_dir, tag):
        """
        Copy the stdout and stderr files that the aggregator and party jobs produce back to the
        local automator machine. Uses the passed-in paramiko client.

        :param machine_staging_dir: staging dir on agg/party machine
        :type machine_staging_dir: `string`
        :param client: the remote machine client to use to obtain the logs
        :type client: paramiko.client.SSHClient
        :param local_staging_dir: staging dir on automator machine
        :type local_staging_dir: `string`
        :param tag: the agg/party string corresponding to the client process (i.e. 'agg', 'party0')
        :type tag: `string`
        :return: None
        """
        if local_staging_dir == machine_staging_dir:
            return
        Runner.__copy_from_server(
            f"{machine_staging_dir}/stdout_{tag}.txt", client, f"{local_staging_dir}/stdout_{tag}.txt"
        )
        Runner.__copy_from_server(
            f"{machine_staging_dir}/stderr_{tag}.txt", client, f"{local_staging_dir}/stderr_{tag}.txt"
        )

    def get_metrics_filepath(self):
        """
        Extract the templated filepath to the metrics file on the automator's machine;
        used throughout to ensure a consistent location is used for each trials' metrics files.
        Leverages instance variables that store the party configs for the current trial.

        :param: None
        :return: A template for the the absolute paths to the metrics files for the current trial
        :rtype: `str`
        """
        return "{}/{}/trial{}/{}.{}".format(
            self.__exp_info["local_staging_dir"],
            "${ts}",
            "${trial}",
            Path(self.__config_party_dicts[0]["metrics_recorder"]["output_file"]).name,
            self.__config_party_dicts[0]["metrics_recorder"]["output_type"],
        ).replace("party0", "party${id}")

    def __copy_metrics_to_local(self, party_clients):
        """
        Copy the metrics files that the party jobs produce back to the local automator machine.
        Uses the passed-in paramiko clients.

        :param party_clients: the remote machine client to use to obtain the logs
        :type party_clients: `list[paramiko.client.SSHClient]`
        :return: None
        """
        # copy stuff back
        metrics_file_exists = [True for p in range(self.__exp_info["n_parties"])]
        for pi, config_party_dict in enumerate(self.__config_party_dicts):
            if "metrics_recorder" not in config_party_dict:
                metrics_file_exists[pi] = False
                continue

            metrics_output_filepath_remote = "{}.{}".format(
                config_party_dict["metrics_recorder"]["output_file"],
                config_party_dict["metrics_recorder"]["output_type"],
            )

            metrics_output_filepath_local = Template(self.get_metrics_filepath()).substitute(
                {"ts": self.__exp_timestamp, "trial": self.__trial_cur, "id": pi}
            )

            # don't copy files if there was no metrics recording
            metrics_file_exists[pi] = Runner.__stat_on_server(party_clients[pi], metrics_output_filepath_remote)
            if not metrics_file_exists[pi]:
                print("No metrics file.")
                continue

            Runner.__copy_from_server(metrics_output_filepath_remote, party_clients[pi], metrics_output_filepath_local)

            print(f"Wrote output data to {metrics_output_filepath_local}")

        return metrics_file_exists

    @staticmethod
    def stage_trial_files(
        ibmfl_dir,
        generated_files_dir,
        local_trial_dir,
        machine_trial_dir,
        config_agg_dict=None,
        config_party_dicts=None,
    ):
        """
        - Copy all files placed into generated_files_dir by the IBMFL generate_* scripts and place
          them all flat into local_trial_dir.
        - Update the paths inside the configs using machine_trial_dir, assuming that they will be
          copied there before the agg and party procs are started.
        - Use the agg and party configs inside generated_files_dir by default, or use the agg &
          party configs inside the optional final two arguments, if they are passed and not None.
        - Copy the dataset using ibmfl_dir (since the scripts don't allow configuration of that
          output path as of now).
        - Return a dictionary with the keys corresponding to the procs ('agg', 'partyX') whose
          values are lists of all the files needed for each of those procs.

        :param ibmfl_dir: the directory where the generate_configs.py and generate_data.py
        scripts live
        :type ibmfl_dir: `str`
        :param generated_files_dir: the directory passed to generate_*.py scripts; where they
        generated the ./data and ./configs folders to place their output
        :type ibmfl_dir: `str`
        :param local_trial_dir: where you want the files to be copied to
        :type local_trial_dir: `str`
        :param machine_trial_dir: where you'll place the files before the run, to update the
        configs; just specify "local_trial_dir" if you don't plan to move them again
        :type machine_trial_dir: `str`
        :param config_agg_dict: if you want to edit the agg config before calling this function,
        parse it into a dictionary and pass it here
        :type config_agg_dict: `dict`
        :param config_party_dicts: if you want to edit the party  configs before calling this
        function, parse them into dictionaries and pass them here in a list (ordered by party id)
        :type config_party_dicts: `list[dict]`
        :return: a dictionary with key for each process, listing the paths to the files it needs
        :rtype: `dict{str,list}`
        """
        import re
        from collections import MutableMapping
        from functools import reduce
        from operator import getitem

        # flattens a dictionary
        def flatten(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, MutableMapping):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        # accesses a value in a nested dictionary with a list of keys
        def dict_set_nested(nested_dict, key_list, value):
            reduce(getitem, key_list[:-1], nested_dict)[key_list[-1]] = value

        # convert all our input strings into path objects
        ibmfl_pobj = Path(ibmfl_dir)
        generated_files_pobj = Path(generated_files_dir)
        local_trial_pobj = Path(local_trial_dir)
        machine_trial_pobj = Path(machine_trial_dir)

        # get all the files in our relevant folders
        generated_files = (
            tuple()
            + tuple(Path(f"{generated_files_dir}/configs").rglob("*.*"))
            + tuple(Path(f"{generated_files_dir}/data").rglob("*.*"))
        )

        proc_file_map = {}

        # if the file is a config, open it and handle the filepaths in it
        for file in generated_files:
            if "config_" in str(file):
                proc_label = re.search("config_(.*).yml", str(file)).group(1)
                if config_agg_dict is not None and "agg" in proc_label:
                    orig_config = config_agg_dict
                elif config_party_dicts is not None and "party" in proc_label:
                    orig_config = config_party_dicts[int(proc_label[-1])]
                else:
                    with open(file, "r") as stream:
                        orig_config = yaml.load(stream.read(), Loader=yaml.Loader)
                flat_config = flatten(orig_config)
                proc_file_map[proc_label] = []
            else:
                continue
            for k, v in flat_config.items():
                # determine if this entry contains a filepath we need to handle
                if isinstance(v, str):
                    v_pobj = Path(v)
                    if str(generated_files_pobj) in str(v_pobj):
                        g_filepath = v_pobj
                    elif "examples/datasets" in str(v_pobj):
                        g_filepath = ibmfl_pobj.joinpath(v_pobj)
                    else:
                        continue
                else:
                    continue
                l_filepath = local_trial_pobj.joinpath(g_filepath.name)
                m_filepath = machine_trial_pobj.joinpath(g_filepath.name)
                # copy the file/dir to our local trial dir
                if not l_filepath.is_file() and "output" not in k:
                    from shutil import copyfile, copytree

                    if g_filepath.is_dir():
                        try:
                            copytree(g_filepath, l_filepath)
                        except IOError:
                            # print('Warn: IOError when copying {} to {}. Perhaps dir already exists'.format(g_filepath,
                            # l_filepath))
                            # suppress error if directory exists
                            pass
                    else:
                        copyfile(g_filepath, l_filepath)
                # don't plan to scp it to the machine if it's an output file
                if "output" not in k:
                    proc_file_map[proc_label] += [l_filepath]
                # set the path in the config to the machine trial dir, where the run happens
                dict_set_nested(orig_config, k.split("."), str(m_filepath))

            with open(f"{local_trial_dir}/{file.name}", "w") as local_trial_config_file:
                yaml.dump(orig_config, local_trial_config_file)

        return proc_file_map

    def run_trial(self, trial_info, config_agg, config_parties, ts, ui_mode="nb"):
        """
        Run a trial, the thing we vary are the config dicts

        :param trial_info: trial data
        :type trial_info: `dict`
        :param config_agg: info for aggregator machine as per either a config_agg.yml file
        (already filled via e.g. the user or the notebook UI) or via the `config_agg_tmpl.yml` file
        :type config_agg: `dict`
        :param config_parties: info for party machines as per either a list of config_partyN.yml files
        (already filled via e.g. the user or the notebook UI) or via the `config_party_tmpl.yml` file
        :type config_parties: `list[dict]`
        :param ts: pseudo-unique generated timestamp for this trial
        :type ts: `str`
        :param ui_mode: tells us whether we're triggering the automator from the notebook or CLI
        :type ui_mode: `str`
        :return: None
        """
        # fill this trial's config template with actual values

        # > agg
        if ui_mode == "cli":
            config_agg_str = Template(config_agg).substitute(
                {
                    "agg_ip": trial_info["agg_machine"]["ip_address"],
                    "agg_port": trial_info["agg_machine"]["port_number"],
                    "staging_dir": trial_info["agg_machine"]["staging_dir"],
                    "ibmfl_dir": trial_info["agg_machine"]["ibmfl_dir"],
                    "n_rounds": trial_info["n_rounds"],
                    "n_parties": trial_info["n_parties"],
                    "ts": ts,
                }
            )
        elif ui_mode == "nb":
            config_agg_str = Template(config_agg).safe_substitute({"ts": ts})
        else:
            print("Bad ui mode.")
            exit(0)
        config_agg_dict = yaml.load(config_agg_str, Loader=yaml.Loader)

        # > parties
        self.__config_party_dicts = []
        for pi in range(trial_info["n_parties"]):
            if ui_mode == "cli":
                config_party_str = Template(config_parties).substitute(
                    {
                        "agg_ip": trial_info["agg_machine"]["ip_address"],
                        "agg_port": trial_info["agg_machine"]["port_number"],
                        "party_ip": trial_info["party_machines"][pi]["ip_address"],
                        "party_port": trial_info["party_machines"][pi]["port_number"],
                        "staging_dir": trial_info["party_machines"][pi]["staging_dir"],
                        "id": pi,
                        "ts": ts,
                    }
                )
            elif ui_mode == "nb":
                config_party_str = Template(config_parties[pi]).safe_substitute({"ts": ts})
            else:
                print("Bad ui mode.")
                exit(0)
            self.__config_party_dicts += [yaml.load(config_party_str, Loader=yaml.Loader)]

        print("Trial has ID {}".format(ts))
        generated_files_dir = str(Path(trial_info["local_staging_dir"]).joinpath(self.__exp_timestamp))
        local_trial_dir = Path(f'{trial_info["local_staging_dir"]}/{self.__exp_timestamp}/trial{self.__trial_cur}')
        local_trial_dir.mkdir(parents=True, exist_ok=True)
        staging_trial_dir = "{}/{}".format(trial_info["agg_machine"]["staging_dir"], ts)
        Runner.__dump_trial_metadata(
            trial_info, config_agg_dict, self.__config_party_dicts, self.__exp_timestamp, self.__trial_cur
        )

        # get key variables from config
        n_rounds = config_agg_dict["hyperparams"]["global"]["rounds"]
        n_parties = config_agg_dict["hyperparams"]["global"]["num_parties"]

        if ui_mode == "nb":
            # copy to trial dir
            proc_file_dict = Runner.stage_trial_files(
                trial_info["local_ibmfl_dir"],
                generated_files_dir,
                local_trial_dir,
                staging_trial_dir,
                config_agg_dict,
                self.__config_party_dicts,
            )
        elif ui_mode == "cli":
            proc_labels = ["agg"] + [f"party{pi}" for pi in range(trial_info["n_parties"])]
            proc_file_dict = {proc_label: [] for proc_label in proc_labels}

        # start agg
        agg_client, agg_handles = self.__start_agg_job(
            config_agg_dict, proc_file_dict["agg"], trial_info, self.__trial_cur, ts
        )
        agg_handles[1].channel.recv(1024).decode()
        print("Started agg...")

        # start parties
        party_clients = []
        party_handles_list = []
        for pi, config_party_dict in enumerate(self.__config_party_dicts):
            party_client, party_handles = self.__start_party_job(
                config_party_dict, proc_file_dict[f"party{pi}"], trial_info, self.__trial_cur, pi, ts
            )
            print(f"Started party {pi}...")
            # keep party variables
            party_clients += [party_client]
            party_handles_list += [party_handles]

        print("All jobs started.")

        # wait till we're done
        exit_statuses = []
        for party_handles in party_handles_list:
            exit_statuses += [party_handles[1].channel.exit_status_ready()]
        exit_statuses += [agg_handles[1].channel.exit_status_ready()]

        round_progress = tqdm(total=n_rounds, desc="Round progress: ")
        party_progress = tqdm(total=n_parties, desc="Party responses: ")
        while not all(exit_statuses):
            # display status
            self.__display_status(
                exit_statuses,
                agg_handles[1].channel,
                self.__config_party_dicts,
                round_progress,
                party_progress,
                n_parties,
            )
            # check parties' exit statuses
            for pi, party_handles in enumerate(party_handles_list):
                exit_statuses[pi] = party_handles[1].channel.exit_status_ready()
            # check agg exit status
            exit_statuses[-1] = agg_handles[1].channel.exit_status_ready()
            # don't go too fast
            time.sleep(1)

        # print final agg/party exit status
        sys.stdout.write(
            "\rAgg finished: {} & Parties finished: {}\n".format(
                "yes" if exit_statuses[-1] else "no", [vi for vi, v in enumerate(exit_statuses[:-1]) if v]
            )
        )
        sys.stdout.flush()

        # copy back logs and metrics
        machine_staging_dir = f"{trial_info['party_machines'][pi]['staging_dir']}/{ts}"
        local_staging_dir = f"{trial_info['local_staging_dir']}/{self.__exp_timestamp}/trial{self.__trial_cur}"
        Runner.__copy_logs_to_local(machine_staging_dir, agg_client, local_staging_dir, "agg")
        for pi in range(trial_info["n_parties"]):
            Runner.__copy_logs_to_local(machine_staging_dir, party_clients[pi], local_staging_dir, f"party{pi}")
        self.__copy_metrics_to_local(party_clients)

        # close up
        agg_client.close()
        for party_client in party_clients:
            party_client.close()

        print("Trial completed.")

    def convert_machine_dict_from_nb_to_cli(self, machines):
        """
        Helper function for converting the slightly-different format for specifying the machines
        that the notebook uses into the format that the automator expects (same as the CLI mode)

        :param machines: list of machines that are available for use during the IBMFL runs
        :type machines: `dict`
        :return: None
        """
        machine_keys = [
            "ssh_username",
            "ip_address",
            "port_number",
            "staging_dir",
            "ibmfl_dir",
            "venv_dir",
            "venv_uses_conda",
        ]
        if "default" not in machines:
            machines["default"] = {}
        for k in machine_keys:
            if k in machines:
                machines["default"][k] = machines[k]
                del machines[k]

    def run_experiment(self, exp_info, machines, config_agg, config_parties, ui_mode="nb", ts=None):
        """
        Run an experiment by generating explicit configs from the template configs provided

        :param exp_info: experiment data
        :type exp_info: `dict`
        :param machines: list of machines that are available for use during the IBMFL runs
        for this experiment; a shared machine list can be specified for multiple experiments,
        and machines can be shuffled for the trials of a given experiment
        :type machines: `dict`
        :param config_agg: info for aggregator machine as per either a config_agg.yml file
        (already filled via e.g. the user or the notebook UI) or via the `config_agg_tmpl.yml` file
        :type config_agg: `dict`
        :param config_parties: info for party machines as per either a list of config_partyN.yml files
        (already filled via e.g. the user or the notebook UI) or via the `config_party_tmpl.yml` file
        :type config_parties: `list[dict]`
        :param ui_mode: tells us whether we're triggering the automator from the notebook or CLI
        :type ui_mode: `str`
        :param ts: timestamp to use as a label for this experiment
        :type ts: `str`
        :return: None
        """
        if ts is None:
            self.__exp_timestamp = Runner.generate_timestamp()
        else:
            self.__exp_timestamp = ts
        # pp.pprint(exp_info)
        n_trials = exp_info["n_trials"]
        n_parties = exp_info["n_parties"]
        self.__exp_info = exp_info
        exp_path = Path(f"{exp_info['local_staging_dir']}/{self.__exp_timestamp}")
        exp_path.mkdir(parents=True, exist_ok=True)
        exp_path_latest = Path(f"{exp_info['local_staging_dir']}/latest")
        if exp_path_latest.is_symlink():
            exp_path_latest.unlink()
        exp_path_latest.symlink_to(exp_path)
        # run trials
        for ti in range(n_trials):
            self.__trial_cur = ti + 1
            trial_info = self.get_trial_info(exp_info, machines, ti)
            ts_obj = datetime.now(timezone.utc)
            ts_fname = ts_obj.strftime("%Y%m%dT%H%M%S")
            ts_print = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
            print("Starting trial {}/{} at {}:".format(ti + 1, n_trials, ts_print))
            self.run_trial(trial_info, config_agg, config_parties, ts_fname, ui_mode)

    def get_experiment_output(self):
        """
        Return a dictionary with the data for the whole experiment available, also adding more
        usable versions of the raw timing values that the IBMFL metrics module produces.

        :param: None
        :return: The metrics dictionary containing data for all trials and parties of experiment
        :rtype: `dict[list[list[np.array]]]`
        """
        # TODO: explicitly handle the case where postproc is requested but metrics don't exist
        # return an empty metrics dictionary if we had no metrics
        # if not any(metrics_file_exists):
        #    print('No metrics output for this experiment.')
        #    return None

        # otherwise, load the metrics files and parse them
        dat = ibmfl_postproc.parse_party_data(self.get_metrics_filepath(), 1, 2)
        offset_methods = {"off": ibmfl_postproc.offset_method_first, "del": ibmfl_postproc.offset_method_delta}
        dat = ibmfl_postproc.offset_vals(dat, [k for k, v in dat.items() if ":ts" in k], offset_methods)

        return dat

    def get_postproc_fn(self):
        """
        Return a handle to the postprocessing function as requested by the experiment config.

        :param: None
        :return: the postprocessing function (should have a specific signature as per the README)
        :rtype: function handle
        """
        return getattr(
            sys.modules["experiment_manager.ibmfl_cli_automator.postprocess"], self.__exp_info["postproc_fn"]
        )

    def call_postproc_fn(self):
        """
        Call the postprocessing function as specified in the experiment config file, using the
        configuration for the ongoing experiment, via self.__exp_info. Produces a plot, which is
        displayed to the user for interactive inspection, saving, or anything else.

        :param: None
        :return: None
        """
        # TODO: explicitly handle the case where postproc is requested but metrics don't exist
        # if not any(metrics_file_exists):
        #    print('No metrics output for this experiment.')
        #    return None
        if "ts" in self.__exp_info["postproc_x_key"]:
            x_axis_val = "time"
        elif "round" in self.__exp_info["postproc_x_key"]:
            x_axis_val = "round"
        else:
            x_axis_val = self.__exp_info["postproc_x_key"]
        self.get_postproc_fn()(
            self.get_metrics_filepath(),
            self.__exp_info["n_trials"],
            self.__exp_info["n_parties"],
            self.__exp_info["postproc_y_keys"],
            x_axis_val,
            self.__exp_info["postproc_x_key"],
        )


if __name__ == "__main__":
    """
    We loops over experiments and trials:
        1) appropriately generate the config templates as needed;
        2) call run_experiment for each FL run (i.e. trial)
        3) call the postprocessing function (if requested via the config) for each experiment
    """
    with open(f"{sys.argv[1]}/config_runner.yml") as config_global_file:
        config_global = yaml.load(config_global_file.read(), Loader=yaml.Loader)
    machines = config_global["machines"]
    experiments = config_global["experiments"]

    exp_runner = Runner()

    for exp_info in experiments:
        with open(f"{sys.argv[1]}/config_agg_tmpl.yml", "r") as config_agg_file:
            config_agg_str = config_agg_file.read()
        with open(f"{sys.argv[1]}/config_party_tmpl.yml", "r") as config_party_file:
            config_party_str = config_party_file.read()
        exp_runner.run_experiment(exp_info, machines, config_agg_str, config_party_str, ui_mode="cli")
        exp_runner.call_postproc_fn()

        # metrics_dict = exp_runner.get_experiment_output()

        # do experiment-specific postprocessing
        # if metrics_dict:
        #    pp.pprint(metrics_dict)
        #    ibmfl_postproc.plot_reward_vs_time(metrics_dict,
        #                                       ['post_train:eval:loss',
        #                                        'post_train:eval:acc',
        #                                        'post_train:eval:precision weighted',
        #                                        'post_train:eval:recall weighted'])
