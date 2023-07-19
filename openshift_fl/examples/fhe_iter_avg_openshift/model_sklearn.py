import os

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier


def get_hyperparams():
    local_params = {"training": {"max_iter": 2}}
    return local_params


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):
    if is_agg:
        return None

    model = SGDClassifier(loss="log", penalty="l2")
    if dataset == "adult":
        model.classes_ = np.array([0, 1])
    elif dataset == "mnist":
        model.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    fname = os.path.join(folder_configs, "model_architecture.pickle")

    with open(fname, "wb") as f:
        joblib.dump(model, f)

    # Generate model spec:
    spec = {"model_definition": fname}

    model = {"name": "SklearnSGDFLModel", "path": "ibmfl.model.sklearn_SGD_linear_fl_model", "spec": spec}

    return model
