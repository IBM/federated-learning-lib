"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2023 All Rights Reserved.
"""
import sys

sys.path.append("../")
import json
import os
import subprocess

import pandas as pd
import yaml

import experiment_manager.ibmfl_cli_automator.run as ibmfl_runner


class ConfigManager:
    """
    The ConfigManager class contains all non-UI logic from the dashboard, necessary to populate objects necessary
    for invoking the runner module.
    """

    def __init__(self):
        self.file_for_supported_combinations = "supported_models.csv"
        self.file_for_hyperparams = "hyperparams_to_models_map.json"

        # dict to store choices made via Notebook UI
        self.nb_config = {"split": {}}
        # set defaults
        self.nb_config["split"]["ppp"] = 100
        self.nb_config["split"]["method"] = "Uniform Random Sampling"
        self.nb_config["parties"] = 5
        self.nb_config["quorum"] = 1
        self.nb_config["record_metrics"] = False

        # Store all supported datasets, models and algorithms in a pandas dataframe
        self.df = pd.read_csv(
            filepath_or_buffer=self.file_for_supported_combinations,
            header=0,
            names=["fusion_identifier", "fusion_algo", "dataset", "model_spec_name", "fl_model", "model_ui"],
            skipinitialspace=True,
        )
        self.df_hyperparams = pd.read_json(path_or_buf=self.file_for_hyperparams)

        self.uimodel_modelid_dict = {
            "Keras": "keras",
            "PyTorch": "pytorch",
            "TensorFlow": "tf",
            "Scikit-learn": "sklearn",
            "None": "None",
        }

        # dict to store details such as machines to run on, paths etc
        self.run_details = {}

        self.exp_runner = ibmfl_runner.Runner()

    def generate_update_configs(self):
        # Get timestamp and add it to the given local staging directory:
        self.nb_config["timestamp_str"] = self.exp_runner.generate_timestamp()
        trial_dir = self.run_details["experiments"][0]["local_staging_dir"] + "/" + self.nb_config["timestamp_str"]

        # Create the staging_directory:
        mkdir_cmd = "mkdir -p " + trial_dir
        process = subprocess.run(mkdir_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            print("Erred: ", process.stderr)
            return None, None

        if "custom_data" in self.nb_config:
            self.move_uploaded_files_to_trial_dir(trial_dir)

        if "custom_data" not in self.nb_config:
            # Generate Data
            print("Generating Data...")

            cmd_to_run = (
                "cd ../; python3 examples/generate_data.py --num_parties "
                + str(self.nb_config["parties"])
                + " -d "
                + self.nb_config["dataset"]
                + " -pp "
                + str(self.nb_config["split"]["ppp"])
                + " -p "
                + trial_dir
            )  # there's only one trial for now
            if "Stratified" in self.nb_config["split"]["method"]:
                cmd_to_run = cmd_to_run + " --stratify"

            # print('Executing {}'.format(cmd_to_run))
            process = subprocess.run(cmd_to_run, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if process.returncode != 0:
                print("Erred: ", process.stderr)
                return None, None

            # path to get datasets from
            data_path = str(process.stdout).split("Data saved in")[-1].strip().replace("\\n'", "")
            print("Data files saved to: {}".format(data_path))

        # Generate Configs:
        print("Generating Configs...")
        if "crypto" in self.nb_config["fusion_identifier"]:
            # if it has either of crypto keras or crypto_multiclass_keras, we need -crypto flags:
            # Todo: Need to let user pick one of {Paillier, ThresholdPaillier}
            cmd_to_run = (
                "cd ../; python3 examples/generate_configs.py"
                + " --num_parties "
                + str(self.nb_config["parties"])
                + " -f "
                + self.nb_config["fusion_identifier"]
                + " -m "
                + self.uimodel_modelid_dict[self.nb_config["model"]]
                + " -crypto Paillier"
                + " --config_path "
                + trial_dir
            )  # there's only one trial for now
        else:
            cmd_to_run = (
                "cd ../; python3 examples/generate_configs.py"
                + " --num_parties "
                + str(self.nb_config["parties"])
                + " -f "
                + self.nb_config["fusion_identifier"]
                + " -m "
                + self.uimodel_modelid_dict[self.nb_config["model"]]
                + " --config_path "
                + trial_dir
            )  # there's only one trial for now

        # add -d and -p flags accordingly
        if "custom_data" in self.nb_config:
            cmd_to_run = cmd_to_run + ' -d custom_dataset -p "" '  # we replace the path down below anyway
        else:
            cmd_to_run = cmd_to_run + " -d " + self.nb_config["dataset"] + " -p " + data_path

        # print('Executing {}'.format(cmd_to_run))
        process = subprocess.run(
            cmd_to_run, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        if process.returncode == 0:
            # save agg and party configs path
            configs_path = os.path.dirname(process.stdout.split("\n")[0].split(":")[1].strip())
            path_to_save_agg_configs = configs_path + "/config_agg.yml"
            print("Aggregator configs saved to: {}".format(path_to_save_agg_configs))
            path_to_save_party_configs = configs_path + "/config_party*.yml"
            print("Party configs saved to: {}".format(path_to_save_party_configs))
        else:
            print("Erred: ", process.stderr)
            return None, None

        # modify hyperparameter text to fix quotes
        hyp_text = self.nb_config["global"]
        # Python uses True/False, while JSON does true/false
        if "True" in hyp_text:
            hyp_text = json.loads(hyp_text.replace("'", '"').replace("True", "true"))
        elif "False" in hyp_text:
            hyp_text = json.loads(hyp_text.replace("'", '"').replace("False", "false"))
        else:
            hyp_text = json.loads(hyp_text.replace("'", '"'))
        if "plus" in self.nb_config["fusion_identifier"]:
            rho = hyp_text["rho"]
        self.nb_config["global"] = hyp_text

        if "local" in self.nb_config.keys():
            hyp_text = self.nb_config["local"]
            # Python uses True/False, while JSON does true/false
            if "True" in hyp_text:
                hyp_text = json.loads(hyp_text.replace("'", '"').replace("True", "true"))
            elif "False" in hyp_text:
                hyp_text = json.loads(hyp_text.replace("'", '"').replace("False", "false"))
            else:
                hyp_text = json.loads(hyp_text.replace("'", '"'))

            if "plus" in self.nb_config["fusion_identifier"]:
                alpha = hyp_text["training"].pop("alpha")
            self.nb_config["local"] = hyp_text

        # add num_parties as a key under global, to match the structure in the agg yaml configs
        val = self.nb_config.pop("parties")
        self.nb_config["global"]["num_parties"] = val
        val = self.nb_config.pop("quorum")
        self.nb_config["global"]["perc_quorum"] = val

        # Load Aggregator Config
        with open(path_to_save_agg_configs, "r") as stream:
            try:
                agg_config = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)
                return None, None

        # for local runs, update the dirs to all the "machines" (they're all local)
        if self.run_details["isLocalRun"]:
            self.run_details["machines"]["ibmfl_dir"] = self.run_details["experiments"][0]["local_ibmfl_dir"]
            self.run_details["machines"]["staging_dir"] = self.run_details["experiments"][0]["local_staging_dir"]

        # Modify aggregator config with values captured from the UI:
        # - update the hyperparameters object with newer global and local objects as updated above
        # - update ip and port from the run_details object
        # - update data handler path to reflect custom datahandler code, if chosen/provided
        agg_config["hyperparams"]["global"] = self.nb_config["global"]
        if "local" in self.nb_config.keys():
            agg_config["hyperparams"]["local"] = self.nb_config["local"]
        agg_machine = self.run_details["experiments"][0]["agg_machine"]  # there's only one trial for now

        if not self.run_details["isLocalRun"]:
            agg_config["connection"]["info"]["ip"] = self.run_details["machines"][agg_machine]["ip_address"]
            agg_config["connection"]["info"]["port"] = int(self.run_details["machines"][agg_machine]["port_number"])
            # Todo: support custom dataset for remote runs
        else:
            self.run_details["machines"][agg_machine]["ip_address"] = agg_config["connection"]["info"]["ip"]
            self.run_details["machines"][agg_machine]["port_number"] = agg_config["connection"]["info"]["port"]
            self.run_details["machines"][agg_machine]["ssh_username"] = os.getenv("USER")

        if "custom_model" in self.nb_config and "model" in agg_config:
            dst = self.move_model_file_to_trial_dir(agg_config)
            agg_config["model"]["spec"]["model_definition"] = dst

        # Write this updated yaml to file
        with open(path_to_save_agg_configs, "w") as out:
            yaml.safe_dump(agg_config, out, default_flow_style=False)
        print("Updated Aggregator config at {}".format(path_to_save_agg_configs))

        # Modify party config with values accepted from the UI
        # - update IP address, port for agg and party as received from the UI (only remote runs)
        # - add metrics section (both remote and local run) -- if needed
        # - add alpha, if model chosen is Fed+
        # - update data handler path to reflect custom datahandler code, if chosen/provided
        if not self.run_details["isLocalRun"]:
            currParty = 0
            for eachMachine in self.run_details["experiments"][0]["party_machines"]:  # there's only one trial for now
                # Load
                with open(path_to_save_party_configs.replace("*", str(currParty))) as stream:
                    try:
                        party_config = yaml.safe_load(stream)
                    except yaml.YAMLError as e:
                        print(e)
                        return None, None

                agg_machine = self.run_details["experiments"][0]["agg_machine"]  # there's only one trial for now
                # Modify
                party_config["aggregator"]["ip"] = self.run_details["machines"][agg_machine]["ip_address"]
                party_config["aggregator"]["port"] = self.run_details["machines"][agg_machine]["port_number"]

                party_config["connection"]["info"]["ip"] = self.run_details["machines"][eachMachine]["ip_address"]
                party_config["connection"]["info"]["port"] = int(
                    self.run_details["machines"][eachMachine]["port_number"]
                )
                party_config["connection"]["info"]["port"] = int(
                    self.run_details["machines"][eachMachine]["port_number"]
                )
                # Todo: DRY!
                if self.nb_config["record_metrics"]:
                    # Metrics section to add to each party config
                    party_config["metrics_recorder"] = {}
                    party_config["metrics_recorder"]["name"] = "MetricsRecorder"
                    party_config["metrics_recorder"]["path"] = "ibmfl.party.metrics.metrics_recorder"
                    party_config["metrics_recorder"]["output_file"] = "${config_dir}/metrics_party${id}".replace(
                        "${config_dir}", self.run_details["machines"][eachMachine]["staging_dir"]
                    ).replace("${id}", str(currParty))
                    party_config["metrics_recorder"]["output_type"] = "json"
                    party_config["metrics_recorder"]["compute_pre_train_eval"] = False
                    party_config["metrics_recorder"]["compute_post_train_eval"] = True

                if self.nb_config["fusion_identifier"] == "fedavgplus":  # Todo: CoMed+ and GeoMed+?
                    party_config["local_training"]["info"]["alpha"] = alpha
                    party_config["local_training"]["info"]["rho"] = rho

                if "custom_data" in self.nb_config.keys():
                    party_config["data"]["name"] = self.nb_config["custom_data"]["name"]
                    party_config["data"]["path"] = self.nb_config["custom_data"]["dh_path"]
                    file_ext = self.nb_config["custom_data"]["data_path"]["party" + str(currParty)].split(".")[-1]
                    if file_ext == "npz":
                        party_config["data"]["info"]["npz_file"] = self.nb_config["custom_data"]["data_path"][
                            "party" + str(currParty)
                        ]
                    else:
                        party_config["data"]["info"]["txt_file"] = self.nb_config["custom_data"]["data_path"][
                            "party" + str(currParty)
                        ]
                if "custom_model" in self.nb_config:
                    # assuming all generated party configs have a model section
                    dst = self.move_model_file_to_trial_dir(party_config)
                    party_config["model"]["spec"]["model_definition"] = dst

                # Finally, write updated party config to file
                with open(path_to_save_party_configs.replace("*", str(currParty)), "w") as out:
                    yaml.safe_dump(party_config, out, default_flow_style=False)
                currParty += 1
                # Todo: support custom dataset for remote runs
        else:
            currParty = 0
            for eachMachine in self.run_details["experiments"][0]["party_machines"]:  # there's only one trial for now
                # Load
                with open(path_to_save_party_configs.replace("*", str(currParty))) as stream:
                    try:
                        party_config = yaml.safe_load(stream)
                    except yaml.YAMLError as e:
                        print(e)
                        return None, None

                # save IP addr and port number from the party config, into `run_details` dict, for runner's use
                self.run_details["machines"][eachMachine]["ip_address"] = party_config["connection"]["info"]["ip"]
                self.run_details["machines"][eachMachine]["port_number"] = party_config["connection"]["info"]["port"]
                self.run_details["machines"][eachMachine]["ssh_username"] = os.getenv("USER")

                if self.nb_config["record_metrics"]:
                    # Metrics section to add to each party config
                    party_config["metrics_recorder"] = {}
                    party_config["metrics_recorder"]["name"] = "MetricsRecorder"
                    party_config["metrics_recorder"]["path"] = "ibmfl.party.metrics.metrics_recorder"
                    party_config["metrics_recorder"]["output_file"] = "${config_dir}/metrics_party${id}".replace(
                        "${config_dir}", trial_dir
                    ).replace("${id}", str(currParty))
                    party_config["metrics_recorder"]["output_type"] = "json"
                    party_config["metrics_recorder"]["compute_pre_train_eval"] = False
                    party_config["metrics_recorder"]["compute_post_train_eval"] = True

                if self.nb_config["fusion_identifier"] == "fedplus":
                    party_config["local_training"]["info"]["alpha"] = alpha

                if "custom_data" in self.nb_config.keys():
                    party_config["data"]["name"] = self.nb_config["custom_data"]["name"]
                    party_config["data"]["path"] = self.nb_config["custom_data"]["dh_path"]
                    file_ext = self.nb_config["custom_data"]["data_path"]["party" + str(currParty)].split(".")[-1]
                    if file_ext == "npz":
                        party_config["data"]["info"]["npz_file"] = self.nb_config["custom_data"]["data_path"][
                            "party" + str(currParty)
                        ]
                    else:
                        party_config["data"]["info"]["txt_file"] = self.nb_config["custom_data"]["data_path"][
                            "party" + str(currParty)
                        ]
                if "custom_model" in self.nb_config:
                    # assuming all generated party configs have a model section
                    dst = self.move_model_file_to_trial_dir(party_config)
                    party_config["model"]["spec"]["model_definition"] = dst

                # Finally, write updated party config to file
                with open(path_to_save_party_configs.replace("*", str(currParty)), "w") as out:
                    yaml.safe_dump(party_config, out, default_flow_style=False)

                currParty += 1

        print("Updated Party configs at {}".format(path_to_save_party_configs))

        self.nb_config["local_conf_dir"] = str(os.path.dirname(path_to_save_agg_configs))

        return path_to_save_agg_configs, path_to_save_party_configs

    def move_uploaded_files_to_trial_dir(self, trial_directory):
        # trial dir was created in caller, so skipping check

        # move provided dataset files:
        dst = os.path.join(trial_directory, "datasets")
        if not os.path.exists(dst):
            os.makedirs(dst)
        for key in self.nb_config["custom_data"]["data_path"]:
            src = self.nb_config["custom_data"]["data_path"][key]
            os.rename(src, os.path.join(dst, src.split("/")[-1]))
            print("Moved {} to {}".format(src, dst))

            # update path in nb_config dict:
            self.nb_config["custom_data"]["data_path"][key] = os.path.join(dst, src.split("/")[-1])
        # move provided datahandler file, as it doesn't get moved by runner
        src_dh = self.nb_config["custom_data"]["dh_path"]
        dst_dh = os.path.join(dst, src_dh.split("/")[-1])
        os.rename(src_dh, dst_dh)
        print("Moved {} to {}".format(src_dh, dst_dh))

        # update path in nb_config dict:
        self.nb_config["custom_data"]["dh_path"] = dst_dh

    def move_model_file_to_trial_dir(self, some_config):
        existing_model_def = some_config["model"]["spec"]["model_definition"]
        if os.path.isdir(existing_model_def):
            # as in the case of TF2
            existing_model_file_path = existing_model_def
            # move user provided model file here
            src = self.nb_config["custom_model"]
            # print('user provided model_file_path:', src)
            # look for assets/, variables/ and saved_model.pb files in this folder
            assets_dir = os.path.join(src, "assets")
            variables_dir = os.path.join(src, "variables")
            model_file_path = os.path.join(src, "saved_model.pb")
            # copy each of them
            from distutils.dir_util import copy_tree

            copy_tree(assets_dir, os.path.join(existing_model_file_path, "assets"))
            copy_tree(variables_dir, os.path.join(existing_model_file_path, "variables"))

            from shutil import copyfile

            copyfile(model_file_path, os.path.join(existing_model_file_path, "saved_model.pb"))
            print("Contents of {} written to {}".format(src, existing_model_file_path))

            # return path for the model file to update respective config (same as before in this case)
            return existing_model_file_path

        else:
            # for .h5, .pt, .pickle files
            existing_model_file_path = existing_model_def[: existing_model_def.rfind("/")]
            # remove existing model file
            if os.path.exists(existing_model_def):
                os.remove(existing_model_def)
            # move user provided model file here
            dst = existing_model_file_path
            # print('dst:', dst)
            src = self.nb_config["custom_model"]
            # print('src:', src)
            dst = os.path.join(dst, src.split("/")[-1])
            from shutil import copyfile

            copyfile(src, dst)
            print("Moved {} to {}".format(src, dst))

            # return new path for the model file to update respective config
            return dst
