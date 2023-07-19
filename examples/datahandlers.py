import os
from pathlib import Path


def get_datahandler_config(dh_name, folder_data, party_id, is_agg):
    path = Path(folder_data)
    staging_dir = ""
    if "data" in path.parts:
        # always holds given how generate_data.py appends "data" before generating datasets
        staging_dir_parts = path.parts[: path.parts.index("data")]
        for folder in staging_dir_parts:
            staging_dir = os.path.join(staging_dir, folder)

    if (
        dh_name == "custom_dataset"
        or dh_name == "custom_dataset_pytorch"
        or dh_name == "custom_dataset_tf"
        or dh_name == "custom_dataset_sklearn"
    ):
        data = {
            "name": "MyDataHandler",  # the datahandler class provided at runtime
            "path": "custom_data_handler.py",
            "info": {},
        }
        if is_agg:
            return None

    elif dh_name == "mnist":
        data = {
            "name": "MnistKerasDataHandler",
            "path": "ibmfl.util.data_handlers.mnist_keras_data_handler",
            "info": {"npz_file": os.path.join(folder_data, "data_party" + str(party_id) + ".npz")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "mnist.npz")):
                data["info"] = {"npz_file": os.path.join(staging_dir, "datasets", "mnist.npz")}
            else:
                data["info"] = {"npz_file": os.path.join("examples", "datasets", "mnist.npz")}

    elif dh_name == "mnist_tf":
        data = {
            "name": "MnistTFDataHandler",
            "path": "ibmfl.util.data_handlers.mnist_keras_data_handler",
            "info": {"npz_file": os.path.join(folder_data, "data_party" + str(party_id) + ".npz")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "mnist.npz")):
                data["info"] = {"npz_file": os.path.join(staging_dir, "datasets", "mnist.npz")}
            else:
                data["info"] = {"npz_file": os.path.join("examples", "datasets", "mnist.npz")}

    elif dh_name == "mnist_dp":
        data = {
            "name": "MnistDPKerasDataHandler",
            "path": "ibmfl.util.data_handlers.mnist_keras_data_handler",
            "info": {"npz_file": os.path.join(folder_data, "data_party" + str(party_id) + ".npz")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "mnist.npz")):
                data["info"] = {"npz_file": os.path.join(staging_dir, "datasets", "mnist.npz")}
            else:
                data["info"] = {"npz_file": os.path.join("examples", "datasets", "mnist.npz")}

    elif dh_name == "mnist_sklearn":
        data = {
            "name": "MnistSklearnDataHandler",
            "path": "ibmfl.util.data_handlers.mnist_sklearn_data_handler",
            "info": {"npz_file": os.path.join(folder_data, "data_party" + str(party_id) + ".npz")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "mnist.npz")):
                data["info"] = {"npz_file": os.path.join(staging_dir, "datasets", "mnist.npz")}
            else:
                data["info"] = {"npz_file": os.path.join("examples", "datasets", "mnist.npz")}

    elif dh_name == "mnist_pytorch":
        data = {
            "name": "MnistPytorchDataHandler",
            "path": "ibmfl.util.data_handlers.mnist_pytorch_data_handler",
            "info": {"npz_file": os.path.join(folder_data, "data_party" + str(party_id) + ".npz")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "mnist.npz")):
                data["info"] = {"npz_file": os.path.join(staging_dir, "datasets", "mnist.npz")}
            else:
                data["info"] = {"npz_file": os.path.join("examples", "datasets", "mnist.npz")}

    elif dh_name == "adult":
        data = {
            "name": "AdultDTDataHandler",
            "path": "ibmfl.util.data_handlers.adult_dt_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            data["info"] = {}

    elif dh_name == "adult_pr":
        data = {
            "name": "AdultPRDataHandler",
            "path": "ibmfl.util.data_handlers.adult_pr_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "adult.data")):
                data["info"] = {"txt_file": os.path.join(staging_dir, "datasets", "adult.data")}
            else:
                data["info"] = {"txt_file": os.path.join("examples", "datasets", "adult.data")}

    elif dh_name == "adult_sklearn":
        data = {
            "name": "AdultSklearnDataHandler",
            "path": "ibmfl.util.data_handlers.adult_sklearn_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            return None

    elif dh_name == "adult_sklearn_grw":
        data = {
            "name": "AdultSklearnDataHandler",
            "path": "ibmfl.util.data_handlers.adult_sklearn_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv"), "epsilon": 1},
        }
        if is_agg:
            return None

    elif dh_name == "nursery":
        data = {
            "name": "NurseryDataHandler",
            "path": "ibmfl.util.data_handlers.nursery_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "nursery.data")):
                data["info"] = {"txt_file": os.path.join(staging_dir, "datasets", "nursery.data")}
            else:
                data["info"] = {"txt_file": os.path.join("examples", "datasets", "nursery.data")}

    elif dh_name == "higgs":
        data = {
            "name": "HiggsDataHandler",
            "path": "ibmfl.util.data_handlers.higgs_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            data["info"] = {}

    elif dh_name == "airline":
        data = {
            "name": "AirlineDataHandler",
            "path": "ibmfl.util.data_handlers.airline_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            data["info"] = {}

    elif dh_name == "diabetes":
        data = {
            "name": "DiabetesDataHandler",
            "path": "ibmfl.util.data_handlers.diabetes_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            data["info"] = {}

    elif dh_name == "binovf":
        data = {
            "name": "BinovfDataHandler",
            "path": "ibmfl.util.data_handlers.binovf_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            data["info"] = {}

    elif dh_name == "multovf":
        data = {
            "name": "MultovfDataHandler",
            "path": "ibmfl.util.data_handlers.multovf_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            data["info"] = {}

    elif dh_name == "multovf_keras":
        data = {
            "name": "MultovfKerasDataHandler",
            "path": "ibmfl.util.data_handlers.multovf_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }

    elif dh_name == "linovf":
        data = {
            "name": "LinovfDataHandler",
            "path": "ibmfl.util.data_handlers.linovf_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            data["info"] = {}

    elif dh_name == "cartpole":
        data = {
            "info": {
                "env_spec": {"env_name": "CartPoleEnv", "env_definition": "ibmfl.util.data_handlers.cartpole_env"}
            },
            "name": "CartpoleEnvDataHandler",
            "path": "ibmfl.util.data_handlers.cartpole_env_data_handler",
        }

    elif dh_name == "pendulum":
        data = {
            "info": {
                "env_spec": {"env_name": "PendulumEnv", "env_definition": "ibmfl.util.data_handlers.pendulum_env"}
            },
            "name": "PendulumEnvDataHandler",
            "path": "ibmfl.util.data_handlers.pendulum_env_data_handler",
        }

    elif dh_name == "femnist":
        data = {
            "name": "FemnistKerasDataHandler",
            "path": "ibmfl.util.data_handlers.femnist_keras_data_handler",
            "info": {"npz_file": os.path.join(folder_data, "data_party" + str(party_id) + ".npz")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "femnist")):
                data["info"] = {"data_folder": os.path.join(staging_dir, "datasets", "femnist")}
            else:
                data["info"] = {"data_folder": os.path.join("examples", "datasets", "femnist")}

    elif dh_name == "cifar10":
        data = {
            "name": "Cifar10KerasDataHandler",
            "path": "ibmfl.util.data_handlers.cifar10_keras_data_handler",
            "info": {"npz_file": os.path.join(folder_data, "data_party" + str(party_id) + ".npz")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "cifar10.npz")):
                data["info"] = {"npz_file": os.path.join(staging_dir, "datasets", "cifar10.npz")}
            else:
                data["info"] = {"data_folder": os.path.join("examples", "datasets", "cifar10.npz")}

    elif dh_name == "cifar10_pytorch":
        data = {
            "name": "Cifar10PytorchDataHandler",
            "path": "ibmfl.util.data_handlers.cifar10_pytorch_data_handler",
            "info": {"npz_file": os.path.join(folder_data, "data_party" + str(party_id) + ".npz")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "cifar10.npz")):
                data["info"] = {"data_folder": os.path.join(staging_dir, "datasets", "cifar10.npz")}
            else:
                data["info"] = {"data_folder": os.path.join("examples", "datasets", "cifar10.npz")}

    elif dh_name == "cifar10_tf":
        data = {
            "name": "Cifar10TFDataHandler",
            "path": "ibmfl.util.data_handlers.cifar10_keras_data_handler",
            "info": {"npz_file": os.path.join(folder_data, "data_party" + str(party_id) + ".npz")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "cifar10.npz")):
                data["info"] = {"data_folder": os.path.join(staging_dir, "datasets", "cifar10.npz")}
            else:
                data["info"] = {"data_folder": os.path.join("examples", "datasets", "cifar10.npz")}

    elif dh_name == "compas_sklearn":
        data = {
            "name": "CompasSklearnDataHandler",
            "path": "ibmfl.util.data_handlers.compas_sklearn_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "compas-scores-two-years.csv")):
                data["info"] = {"txt_file": os.path.join(staging_dir, "datasets", "compas-scores-two-years.csv")}
            else:
                data["info"] = {"txt_file": os.path.join("examples", "datasets", "compas-scores-two-years.csv")}

    elif dh_name == "compas_pr":
        data = {
            "name": "CompasPRDataHandler",
            "path": "ibmfl.util.data_handlers.compas_pr_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "compas-scores-two-years.csv")):
                data["info"] = {"txt_file": os.path.join(staging_dir, "datasets", "compas-scores-two-years.csv")}
            else:
                data["info"] = {"txt_file": os.path.join("examples", "datasets", "compas-scores-two-years.csv")}

    elif dh_name == "compas_sklearn_grw":
        data = {
            "name": "CompasSklearnDataHandler",
            "path": "ibmfl.util.data_handlers.compas_sklearn_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv"), "epsilon": 1},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "compas-scores-two-years.csv")):
                data["info"] = {"txt_file": os.path.join(staging_dir, "datasets", "compas-scores-two-years.csv")}
            else:
                data["info"] = {"txt_file": os.path.join("examples", "datasets", "compas-scores-two-years.csv")}

    elif dh_name == "german_sklearn":
        data = {
            "name": "GermanSklearnDataHandler",
            "path": "ibmfl.util.data_handlers.german_sklearn_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv")},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "german.data")):
                data["info"] = {"txt_file": os.path.join(staging_dir, "datasets", "german.data")}
            else:
                data["info"] = {"txt_file": os.path.join("examples", "datasets", "german.data")}

    elif dh_name == "german_sklearn_grw":
        data = {
            "name": "GermanSklearnDataHandler",
            "path": "ibmfl.util.data_handlers.german_sklearn_data_handler",
            "info": {"txt_file": os.path.join(folder_data, "data_party" + str(party_id) + ".csv"), "epsilon": 1},
        }
        if is_agg:
            if os.path.exists(os.path.join(staging_dir, "datasets", "german.data")):
                data["info"] = {"txt_file": os.path.join(staging_dir, "datasets", "german.data")}
            else:
                data["info"] = {"txt_file": os.path.join("examples", "datasets", "german.data")}

    elif dh_name == "federated-clustering":
        data = {
            "name": "FederatedClusteringDataHandler",
            "path": "ibmfl.util.data_handlers.federated_clustering_data_handler",
            "info": {"npz_file": os.path.join(folder_data, "data_party" + str(party_id) + ".npz")},
        }
        if is_agg:
            data["info"] = {}

    elif dh_name == "wikipedia":
        data = {
            "name": "WikipediaDoc2VecDataHandler",
            "path": "ibmfl.util.data_handlers.wikipedia_doc2vec_data_handler",
            "info": {"pickled_file": os.path.join(folder_data, "data_party" + str(party_id) + ".pickle")},
        }
        if is_agg:
            data["info"] = {"pickled_file": os.path.join("examples", "datasets", "wikipedia.pickle")}

    return data
