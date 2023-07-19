"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2023 All Rights Reserved.
"""
import os
import shutil
import sys

sys.path.append("../")
import ast
import json
from json import JSONDecodeError

import config_manager
import yaml

# widget imports
from IPython.display import display
from ipywidgets import (
    HTML,
    Box,
    Button,
    Dropdown,
    HBox,
    IntSlider,
    Label,
    Layout,
    Output,
    RadioButtons,
    Text,
    Textarea,
    VBox,
)

import experiment_manager.ibmfl_cli_automator.run as ibmfl_runner


class DashboardUI:
    """
    The DashboardUI class contains all widgets required in the dashboard, as well as their event handler methods.
    """

    def __init__(self):
        self.mgr = config_manager.ConfigManager()
        self.exp_runner = ibmfl_runner.Runner()

        self.params_widgets = []
        self.hyperparams_dict = {}

    def generate_model_dataset_fusion_ui(self):
        ui_model_choices = self.mgr.df.model_ui.unique()
        model_header = HTML(
            value="<{size}>Model details".format(size="h4"), layout=Layout(width="auto", grid_area="model_header")
        )

        # Model Selection:
        model_dropdown = Dropdown(
            options=["Choose your model"] + list(ui_model_choices),
            description="Model:",
            disabled=False,
            layout=Layout(width="60%", grid_area="model_dr"),
        )

        def model_dropdown_eventhandler(change):
            model_chosen = change.new
            self.mgr.nb_config["model"] = model_chosen
            # metrics are only supported for (some) Keras models
            if model_chosen == "Keras":
                metrics_or_not.children[1].disabled = False
            else:
                metrics_or_not.children[1].value = "No"
                metrics_or_not.children[1].disabled = True
                self.mgr.nb_config["record_metrics"] = False

        model_dropdown.observe(model_dropdown_eventhandler, names="value")

        dataset_header = HTML(
            value="<{size}>Dataset details".format(size="h4"), layout=Layout(width="auto", grid_area="dataset_header")
        )

        # path for model file
        custom_model_filepath = Text(
            value="",
            placeholder="Paste path to model file (optional)",
            description="Model File:",
            grid_area="custom_model",
        )

        interim_dir = os.getcwd()  # gets moved to staging_dir later
        data_dir = os.path.join(interim_dir, "uploadedFiles")
        if os.path.exists(data_dir):
            shutil.rmtree(path=data_dir)
        os.makedirs(data_dir)

        def upload_model_path_handler(change):
            model_file_path = change.new
            # print(model_file_path + ' received!')
            if not os.path.exists(model_file_path):
                print(model_file_path + " does not exist!")
                return

            if os.path.isfile(model_file_path):
                # .h5, .pt or .pickle files
                filename = model_file_path.split("/")[-1]
                # copy model file to data_dir:
                from shutil import copyfile

                # copy model file to this dir
                copyfile(model_file_path, os.path.join(data_dir, filename))
                print(filename + " written to " + data_dir + "/" + filename)
                self.mgr.nb_config["custom_model"] = os.path.join(data_dir, filename)
            elif os.path.isdir(model_file_path):
                # TF SavedModel format uses a directory
                # copy dir to data_dir:
                assets_dir = os.path.join(model_file_path, "assets")
                variables_dir = os.path.join(model_file_path, "variables")
                model_file = os.path.join(model_file_path, "saved_model.pb")
                dirname = model_file_path.split("/")[-1]
                if os.path.isdir(assets_dir) and os.path.isdir(variables_dir) and os.path.isfile(model_file):
                    # adheres to TF SavedModel format, so we copy
                    from distutils.dir_util import copy_tree

                    copy_tree(model_file_path, os.path.join(data_dir, dirname))
                    print(model_file_path + " written to " + data_dir + "/" + dirname)
                    self.mgr.nb_config["custom_model"] = os.path.join(data_dir, dirname)

        custom_model_filepath.observe(upload_model_path_handler, names="value")

        dataset_dropdown = Dropdown(
            options=["Choose your dataset"],  # + determine_allowed_datasets(),
            description="Dataset:",
            disabled=False,
            layout=Layout(width="80%", grid_area="dataset"),
        )

        def update_supported_datasets(change):
            model_chosen = change.new
            rows_for_model = self.mgr.df[self.mgr.df.model_ui == model_chosen]
            dataset_dropdown.options = list(rows_for_model["dataset"].unique())

        model_dropdown.observe(update_supported_datasets, "value")

        def dataset_dropdown_eventhandler(change):
            dataset_chosen = change.new
            self.mgr.nb_config["dataset"] = dataset_chosen

        dataset_dropdown.observe(dataset_dropdown_eventhandler, names="value")

        # Data Splitting Strategy:
        splitting_dropdown = Box(
            [
                Label(value="Data Split:", layout=Layout(width="auto")),
                Dropdown(
                    options=["Uniform Random Sampling", "Stratified Sampling (per source class)"],
                    disabled=False,
                    layout=Layout(width="auto"),
                    value="Uniform Random Sampling",
                ),
            ],
            grid_area="dataset_spl",
        )

        def splitting_dropdown_eventhandler(change):
            split_chosen = change.new
            self.mgr.nb_config["split"]["method"] = split_chosen

        splitting_dropdown.children[1].observe(splitting_dropdown_eventhandler, names="value")

        # Points per party when splitting data:
        points_slider = Box(
            [
                Label(value="Points from each party:", layout=Layout(width="auto")),
                IntSlider(min=100, max=1000, layout=Layout(width="50%"), value=100),
            ],
            grid_area="ppp",
        )

        def points_slider_eventhandler(change):
            # print(change)
            ppp = change.new
            self.mgr.nb_config["split"]["ppp"] = ppp

        points_slider.children[1].observe(points_slider_eventhandler, names="value")

        # Add choice to bring custom dataset:
        custom_data = Box(
            [
                HTML(value="<{size}>OR".format(size="h4"), layout=Layout(width="25%")),
                HTML(value="<{size}>Custom Dataset?".format(size="h4"), layout=Layout(width="35%")),
                RadioButtons(options=["Yes", "No"], value="No", disabled=False, layout=Layout(width="40%")),
            ],
            layout=Layout(width="100%", height="100%"),
            grid_area="custom_data",
        )

        custom_data_html = HTML(
            value='<{size} style="color:red;">Choosing Yes requires you to provide a custom data handler and party '
            "data files".format(size="h5"),
            layout=Layout(width="auto", grid_area="custom_data_html"),
        )

        def custom_data_handler(change):
            if "custom_data" not in self.mgr.nb_config:
                self.mgr.nb_config["custom_data"] = {}
            custom_data.children[2].disabled = True
            if change.new == "Yes":
                # disable built-in dataset UI widgets, purge their key/value from the config dict
                dataset_dropdown.disabled = True
                splitting_dropdown.children[1].disabled = True
                points_slider.children[1].disabled = True
                self.mgr.nb_config.pop("split", None)
                self.mgr.nb_config.pop("dataset", None)
                dh_path = os.path.join(os.getcwd(), "custom_data_handler.py")
                self.mgr.nb_config["custom_data"]["dh_path"] = dh_path
                # get class name from data handler file:
                as_tree = ast.parse(open(dh_path).read())
                classes = []
                for i in as_tree.body:
                    if isinstance(i, ast.ClassDef):
                        classes.append(i.name)
                if len(classes) == 1:
                    print("Found class {} in the data handler provided!".format(classes[0]))
                    self.mgr.nb_config["custom_data"]["name"] = classes[0]
                else:
                    print(
                        "Found {} class(es) in the data handler provided, expected exactly 1. Aborting!".format(
                            len(classes)
                        )
                    )

            # else:  # no need as the widget is disabled after interaction
            #     dataset_dropdown.disabled = False
            #     splitting_dropdown.children[1].disabled = False
            #     points_slider.children[1].disabled = False
            #
            #     # purge custom_data dict
            #     self.mgr.nb_config.pop('custom_data', None)

        custom_data.children[2].observe(custom_data_handler, "value")

        fusion_dropdown = Box(
            [
                HTML(value="<{size}>Fusion Algorithm".format(size="h4"), layout=Layout(width="auto")),
                Dropdown(options=["Choose your Fusion Algorithm"], disabled=False, layout=Layout(width="auto")),
            ],
            grid_area="fusion_dr",
        )

        def update_potential_fusion_algorithm(change):
            model_chosen = self.mgr.nb_config["model"]
            if "custom_data" in self.mgr.nb_config:
                potential_algo = list(self.mgr.df[(self.mgr.df.model_ui == model_chosen)]["fusion_algo"].unique())
            else:
                dataset_chosen = self.mgr.nb_config["dataset"]
                potential_algo = list(
                    self.mgr.df[(self.mgr.df.model_ui == model_chosen) & (self.mgr.df.dataset == dataset_chosen)][
                        "fusion_algo"
                    ].unique()
                )
            fusion_dropdown.children[1].options = potential_algo

        model_dropdown.observe(update_potential_fusion_algorithm, "value")
        dataset_dropdown.observe(update_potential_fusion_algorithm, "value")

        def fusion_dropdown_eventhandler(change):
            fusion_algo_chosen = change.new
            self.mgr.nb_config["fusion"] = fusion_algo_chosen

        fusion_dropdown.children[1].observe(fusion_dropdown_eventhandler, names="value")

        metrics_or_not = Box(
            [
                HTML(value="<{size}>Record Metrics?".format(size="h4"), layout=Layout(width="45%")),
                RadioButtons(options=["Yes", "No"], value="No", disabled=False, layout=Layout(width="20%")),
                HTML(value="<{size}>May not be supported for all models".format(size="h5"), layout=Layout(width="35%")),
            ],
            layout=Layout(width="100%", height="100%"),
            grid_area="metrics_choice",
        )

        def metrics_choice_handler(change):
            metrics_or_not.children[1].disabled = True
            if change.new == "Yes":
                self.mgr.nb_config["record_metrics"] = True

        metrics_or_not.children[1].observe(metrics_choice_handler, names="value")

        return (
            model_header,
            model_dropdown,
            custom_model_filepath,
            dataset_header,
            dataset_dropdown,
            splitting_dropdown,
            points_slider,
            custom_data,
            custom_data_html,
            fusion_dropdown,
            metrics_or_not,
        )

    def generate_parties_hyperparams_ui(self):
        header_parties = HTML(
            value="<{size}>Participants".format(size="h4"), layout=Layout(width="auto", grid_area="header_parties")
        )

        num_parties = Box(
            [
                Label(value="Number of parties:", layout=Layout(width="auto")),
                IntSlider(min=2, max=100, value=5, layout=Layout(width="50%")),
            ],
            grid_area="parties",
        )

        def num_parties_eventhandler(change):
            # print(change)
            parties = change.new
            self.mgr.nb_config["parties"] = parties

        num_parties.children[1].observe(num_parties_eventhandler, names="value")

        parties_in_quorum = Box(
            [
                Label(value="Number of parties in quorum", layout=Layout(width="auto")),
                IntSlider(min=2, max=5, value=5, layout=Layout(width="50%")),
            ],
            grid_area="parties",
        )

        # quorum can have atmost all parties
        def update_quorum_range(*args):
            parties_in_quorum.children[1].max = num_parties.children[1].value
            parties_in_quorum.children[1].value = num_parties.children[1].value

        num_parties.children[1].observe(update_quorum_range, "value")

        def parties_in_quorum_eventhandler(change):
            # print(change)
            quorum = change.new
            self.mgr.nb_config["quorum"] = round(quorum / float(self.mgr.nb_config["parties"]), 2)

        parties_in_quorum.children[1].observe(parties_in_quorum_eventhandler, names="value")

        header_hyperparams = HTML(
            value="<{size}>Hyperparameters".format(size="h4"),
            layout=Layout(width="auto", grid_area="header_hyperparams"),
        )

        confirmation_box = Box()

        hyperparams_text = Box()

        self.determine_hyperparams()
        self.params_widgets.clear()
        self.generate_hyperparam_ui()
        hyperparams_text.children = self.params_widgets

        def confirmation_button_handler(b):
            b.disabled = True
            b.description = "Hyperparams Saved"
            num_parties.children[1].disabled = True
            parties_in_quorum.children[1].disabled = True
            for i in range(len(hyperparams_text.children)):
                hyperparams_text.children[i].disabled = True

            for widget in self.params_widgets:
                self.mgr.nb_config[widget.description] = widget.value

        confirm_butn = Button(
            description="Confirm Hyperparameters",
            disabled=False,
            button_style="warning",
            tooltip="Saves the hyperparameter changes",
            layout=Layout(width="auto", height="40px"),
        )

        confirmation_box.children = (confirm_butn,)
        [
            confirmation_box.children[i].on_click(confirmation_button_handler)
            for i in range(len(confirmation_box.children))
        ]

        return header_parties, num_parties, parties_in_quorum, header_hyperparams, confirmation_box, hyperparams_text

    def determine_hyperparams(self):
        if "custom_data" in self.mgr.nb_config:
            exp_df = self.mgr.df[
                (self.mgr.df.model_ui == self.mgr.nb_config["model"])
                & (self.mgr.df.fusion_algo == self.mgr.nb_config["fusion"])
            ]
        else:
            exp_df = self.mgr.df[
                (self.mgr.df.model_ui == self.mgr.nb_config["model"])
                & (self.mgr.df.dataset == self.mgr.nb_config["dataset"])
                & (self.mgr.df.fusion_algo == self.mgr.nb_config["fusion"])
            ]
        if len(exp_df) != 1:
            # pick the first matching fusion algorithm
            # print('Found multiple matches, picking the first one')
            firstMatch = exp_df.iloc[0]
            # print(firstMatch)
            self.mgr.nb_config["fusion_identifier"] = firstMatch[0]
        else:
            # print(exp_df)
            self.mgr.nb_config["fusion_identifier"] = list(exp_df.fusion_identifier)[0]

        # print('fusion_id:', self.mgr.nb_config['fusion_identifier'])
        model_hyperparams_key = (
            self.mgr.nb_config["fusion_identifier"] + "_" + self.mgr.uimodel_modelid_dict[self.mgr.nb_config["model"]]
        )  # to get hyperparams from df
        self.hyperparams_dict = self.mgr.df_hyperparams[
            self.mgr.df_hyperparams["model_identifier"] == model_hyperparams_key
        ].hyperparams.values[0]

    def generate_hyperparam_ui(self):
        # every model has at most two keys: global and local:
        # print(self.hyperparams_dict)
        params_dict = self.hyperparams_dict

        def inner_generate_hyperparam_ui(params_dict):
            for key in params_dict:
                if type(params_dict[key]) == "dict":
                    inner_generate_hyperparam_ui(params_dict[key])
                else:
                    self.params_widgets.append(
                        Textarea(
                            description=key,
                            value=str(params_dict[key]),
                            layout=Layout(width="400px", height="100px"),
                            grid_area="hyperparams",
                        )
                    )

        inner_generate_hyperparam_ui(params_dict)

    def generate_local_remote_ui(self):
        local_or_remote = Box(
            [
                HTML(
                    value="<{size}>Run this experiment locally or on remote machines?".format(size="h4"),
                    layout=Layout(width="auto"),
                ),
                Dropdown(
                    options=["Choose your option", "Run Locally", "Run on Remote Machines"],
                    description="",
                    disabled=False,
                    layout=Layout(width="200px"),
                ),
            ]
        )

        def network_details_tracker(change):
            value = change.new
            subkey = change.owner.description.split(":")[0].replace(" ", "_").lower()
            machine_key = change.owner.placeholder.split(" ")[-1]
            # update the run_details dict, depending on whether it already has some details:
            if len(self.mgr.run_details["machines"][machine_key].keys()) == 0:
                temp_dict = {}
                temp_dict[subkey] = value
                self.mgr.run_details["machines"][machine_key] = temp_dict
            else:
                temp_dict = self.mgr.run_details["machines"][machine_key]
                temp_dict[subkey] = value
                self.mgr.run_details["machines"][machine_key] = temp_dict

        def get_IPaddr_port(party_index=None):
            placeholder_suffix = " for machine" + str(party_index)

            ip_addr = Text(value="", placeholder="IP Address" + placeholder_suffix, description="IP Address:")
            port_num = Text(value="", placeholder="Port Number" + placeholder_suffix, description="Port Number:")
            ssh_user = Text(value="", placeholder="ssh username" + placeholder_suffix, description="SSH Username:")

            machine_detail_vbox = VBox(children=[ip_addr, port_num, ssh_user])
            [
                machine_detail_vbox.children[i].observe(network_details_tracker, "value")
                for i in range(len(machine_detail_vbox.children))
            ]
            return machine_detail_vbox

        def path_details_tracker(change):
            value = change.new
            subkey = change.owner.description.split(":")[0].replace(" ", "_").lower()
            if "local" in change.owner.placeholder:
                # this is a local path, put within `experiments` key
                local_subkey = "local_" + subkey
                self.mgr.run_details["experiments"][0][local_subkey] = value  # there's only one trial for now
            else:
                # this is a machine path
                # update the run_details dict, depending on whether it already has some details:
                machine_key = change.owner.placeholder.split(" ")[-1]  # to figure which machine is this for
                if len(self.mgr.run_details["machines"][machine_key].keys()) == 0:
                    temp_dict = {}
                    temp_dict[subkey] = value
                    self.mgr.run_details["machines"][machine_key] = temp_dict
                else:
                    temp_dict = self.mgr.run_details["machines"][machine_key]
                    temp_dict[subkey] = value
                    self.mgr.run_details["machines"][machine_key] = temp_dict

        def get_paths(party_index=None):
            if party_index is None:
                placeholder_suffix = " for local machine"
            else:
                placeholder_suffix = " for machine" + str(party_index)

            config_path = Text(value="", placeholder="Staging Dir" + placeholder_suffix, description="Staging Dir:")
            code_path = Text(value="", placeholder="IBMFL Dir" + placeholder_suffix, description="IBMFL Dir:")

            machine_detail_vbox = VBox(children=[config_path, code_path])
            [
                machine_detail_vbox.children[i].observe(path_details_tracker, "value")
                for i in range(len(machine_detail_vbox.children))
            ]
            return machine_detail_vbox

        networking_deets_box = VBox()

        def venv_box_isConda_handler(change):
            if change.new == "Yes":
                self.mgr.run_details["machines"]["venv_uses_conda"] = True
            else:
                self.mgr.run_details["machines"]["venv_uses_conda"] = False

        def venv_box_venvPath_handler(change):
            self.mgr.run_details["machines"]["venv_dir"] = change.new

        def display_conda_venv_fields():
            venv_box = HBox(
                [
                    RadioButtons(options=["No", "Yes"], description="Use conda?"),
                    Text(
                        value="",
                        placeholder="venv name",
                        description="virtual env:",
                        layout=Layout(width="300px", height="auto"),
                    ),
                ]
            )
            venv_box.children[0].disabled = True  # No support for conda: https://github.com/conda/conda/issues/7980
            venv_box.children[0].observe(venv_box_isConda_handler, "value")
            venv_box.children[1].observe(venv_box_venvPath_handler, "value")
            return venv_box

        def run_details_text_handler(change):
            # print(change.new)
            try:
                self.mgr.run_details = json.loads(change.new)
            except JSONDecodeError:
                if change.new == "":
                    pass
                else:
                    print("Incorrect JSON passed for remote details, check and retry!")

        def machines_dropdown_eventhandler(change):
            # print(change.new)
            agg_machine = change.new.lower()
            self.mgr.run_details["experiments"][0]["agg_machine"] = agg_machine  # there is only one trial for now
            party_machines = []
            for machine in self.mgr.run_details["machines"]:
                party_machines.append(machine)

            # now remove the agg machine from the dict
            party_machines.remove(agg_machine)
            # remove other extra keys if included
            if "venv_dir" in party_machines:
                party_machines.remove("venv_dir")
            if "venv_uses_conda" in party_machines:
                party_machines.remove("venv_uses_conda")
            self.mgr.run_details["experiments"][0]["party_machines"] = party_machines  # there is only one trial for now

        def display_run_details(change):
            change.owner.disabled = True
            self.mgr.run_details["machines"] = {}
            self.mgr.run_details["machines"]["venv_uses_conda"] = False
            self.mgr.run_details["machines"]["venv_dir"] = ".venv"
            self.mgr.run_details["experiments"] = []

            temp_exp_dict = {}
            temp_exp_dict["local_staging_dir"] = ""
            temp_exp_dict["local_ibmfl_dir"] = ""
            conda_fields = display_conda_venv_fields()

            if "Remote" in change.new:
                # remote execution
                # initialise the run_details dictionary
                self.mgr.run_details["isLocalRun"] = False

                temp_exp_dict["agg_machine"] = ""
                temp_exp_dict["party_machines"] = []

                for eachMachine in range(self.mgr.nb_config["parties"] + 1):
                    self.mgr.run_details["machines"]["machine" + str(eachMachine + 1)] = {}
                    self.mgr.run_details["machines"]["machine" + str(eachMachine + 1)]["ip_address"] = ""
                    self.mgr.run_details["machines"]["machine" + str(eachMachine + 1)]["port_number"] = ""
                    self.mgr.run_details["machines"]["machine" + str(eachMachine + 1)]["ssh_username"] = ""
                    self.mgr.run_details["machines"]["machine" + str(eachMachine + 1)]["staging_dir"] = ""
                    self.mgr.run_details["machines"]["machine" + str(eachMachine + 1)]["ibmfl_dir"] = ""

                networking_header_1 = HTML(
                    value="<{size}>Details for remote execution: Fill details into the textbox on the left or in "
                    "individual fields on the right".format(size="h4"),
                    layout=Layout(width="auto"),
                )

                run_details_box = VBox(
                    [
                        Label(value="Machine details:", layout=Layout(width="auto")),
                        Textarea(
                            value=json.dumps(self.mgr.run_details, indent=4),
                            layout=Layout(width="300px", height="700px"),
                        ),
                    ]
                )
                run_details_box.children[1].observe(run_details_text_handler, "value")

                networking_header_2 = HTML(
                    value="<center><{size}>OR".format(size="h3"),
                    layout=Layout(width="auto", margin="5px 15px 5px 15px"),
                )

                all_machines_tuple = ()
                for eachMachine in range(self.mgr.nb_config["parties"] + 1):
                    machine_header = HTML(value="<{size}>Machine{id}".format(size="h4", id=str(eachMachine + 1)))
                    temp_machine_box = VBox()
                    machine_IP = get_IPaddr_port(eachMachine + 1)
                    machine_paths = get_paths(eachMachine + 1)
                    temp_machine_box.children = (machine_header, HBox(children=[machine_IP, machine_paths]))
                    all_machines_tuple = all_machines_tuple + (temp_machine_box,)

                machines_dropdown = Box(
                    [
                        Label(value="Pick machine for running Aggregator:", layout=Layout(width="auto")),
                        Dropdown(
                            options=[""]
                            + ["Machine{id}".format(id=i + 1) for i in range(self.mgr.nb_config["parties"] + 1)],
                            layout=Layout(width="auto"),
                        ),
                    ]
                )

                machines_dropdown.children[1].observe(machines_dropdown_eventhandler, "value")

                temp_local_vbox = VBox()
                local_header = HTML(value="<{size}>Local Directories".format(size="h4"))
                local_path_fields = get_paths()
                temp_local_vbox.children = (local_header, local_path_fields)

                networking_fields_vbox = VBox(layout=Layout(width="auto", border="0.5px solid black"))
                networking_fields_vbox.children = (
                    (conda_fields,)
                    + all_machines_tuple
                    + (
                        machines_dropdown,
                        temp_local_vbox,
                    )
                )
                networking_deets_hbox = HBox(children=[run_details_box, networking_header_2, networking_fields_vbox])
                # save_generate_butn.layout = Layout(width='185px', height='40px', margin='5px 50px 5px 400px')
                networking_deets_box.children = (networking_header_1, networking_deets_hbox)  # , save_generate_butn,)
                self.mgr.run_details["experiments"].append(temp_exp_dict)

            else:
                # local execution
                self.mgr.run_details["isLocalRun"] = True
                temp_exp_dict["agg_machine"] = "local0"
                temp_exp_dict["party_machines"] = [
                    "local{id}".format(id=i + 1) for i in range(self.mgr.nb_config["parties"])
                ]

                # setup dicts to populate IP addr and port number from generated configs later
                self.mgr.run_details["machines"]["local0"] = {}
                for party in temp_exp_dict["party_machines"]:
                    self.mgr.run_details["machines"][party] = {}

                networking_header = HTML(
                    value="<{size}>Details for local execution".format(size="h4"), layout=Layout(width="auto")
                )

                local_paths = get_paths()
                # save_generate_butn.layout = Layout(width='185px', height='40px', margin='5px 50px 5px 50px')
                networking_deets_box.children = (networking_header, conda_fields, local_paths)  # , save_generate_butn)

                self.mgr.run_details["experiments"].append(temp_exp_dict)

        local_or_remote.children[1].observe(display_run_details, "value")

        return (local_or_remote, networking_deets_box)

    def generate_custom_party_data_ui(self):
        def custom_data_filepath_handler(change):
            # print(change)
            party_data_filepath = change.new
            if not os.path.exists(party_data_filepath):
                print(party_data_filepath + " does not exist!")
                return
            party_idx = change["owner"].description.split()[-1][-2]
            filename = party_data_filepath.split("/")[-1]
            # copy model file to data_dir
            from shutil import copyfile

            # copy model file to this dir
            copyfile(party_data_filepath, os.path.join(data_dir, filename))
            print(filename + " written to " + data_dir + "/" + filename)
            self.mgr.nb_config["custom_data"]["data_path"]["party" + str(party_idx)] = os.path.join(data_dir, filename)

        custom_data_paths = []
        interim_dir = os.getcwd()
        data_dir = os.path.join(interim_dir, "uploadedFiles")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.mgr.nb_config["custom_data"]["data_path"] = {}
        # path boxes for party specific files
        for each_party in range(self.mgr.nb_config["parties"]):
            custom_data_filepath = Text(
                value="",
                placeholder="Paste path to dataset file for party",
                description="For party{}:".format(each_party),
            )
            custom_data_filepath.observe(custom_data_filepath_handler, names="value")
            custom_data_paths.append(custom_data_filepath)
        return custom_data_paths

    def generate_display_configs_ui(self):
        def display_configs(agg_conf_path, party_conf_path):
            # Display aggregator and party* configs
            display_header = HTML(value="<{size}>Configs Generated:".format(size="h4"), layout=Layout(width="auto"))

            agg_conf_header = HTML(value="<{size}>Aggregator Config".format(size="h4"), layout=Layout(width="auto"))
            agg_conf = Output(layout={"border": "0.5px solid black"})

            # read agg config from filesystem:
            with open(agg_conf_path) as stream:
                try:
                    agg_config = yaml.safe_load(stream)
                except yaml.YAMLError as e:
                    print(e)

            with agg_conf:
                display(agg_config)

            party_conf_header = HTML(value="<{size}>Party0 Config".format(size="h4"), layout=Layout(width="auto"))
            party_conf = Output(layout={"border": "0.5px solid black"})

            # read party0 from filesystem:
            with open(party_conf_path.replace("*", "0")) as stream:
                try:
                    party_config = yaml.safe_load(stream)
                except yaml.YAMLError as e:
                    print(e)

            # display
            with party_conf:
                display(party_config)

            agg_box = HBox(children=[agg_conf_header, agg_conf], layout=Layout(width="auto", padding="20px"))
            party_box = HBox(children=[party_conf_header, party_conf], layout=Layout(width="auto", padding="10px"))
            party_disclmr_1 = HTML(
                value="<strong><center>Other parties follow config similar to Party0, except connection.info.[ip,port] "
                "and paths",
                layout=Layout(width="auto"),
            )
            party_disclmr_2 = HTML(
                value="<strong><center>Also, each party gets a separate dataset file, split from the chosen dataset",
                layout=Layout(width="auto"),
            )
            config_box.children = [display_header, agg_box, party_box, party_disclmr_1, party_disclmr_2]

        config_ui = Output()
        config_box = VBox(layout=Layout(width="auto"))
        agg_conf_path, party_conf_path = self.mgr.generate_update_configs()
        if agg_conf_path is None or party_conf_path is None:
            print("Error generating configs. Exiting...")
        else:
            display_configs(agg_conf_path, party_conf_path)

        return (config_ui, config_box)
