import os
import pickle

import joblib
import numpy as np
from sklearn.cluster import KMeans


def get_hyperparams():
    local_params = {"training": {"max_iter": 500, "n_clusters": 10}}
    return local_params


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):
    if is_agg:
        return None

    model = KMeans()

    # Save model
    fname = os.path.join(folder_configs, "kmeans-central-model.pickle")
    with open(fname, "wb") as f:
        pickle.dump(model, f)
    # Generate model spec:
    spec = {"model_name": "sklearn-kmeans", "model_definition": fname}

    model = {"name": "SklearnKMeansFLModel", "path": "ibmfl.model.sklearn_kmeans_fl_model", "spec": spec}

    return model
