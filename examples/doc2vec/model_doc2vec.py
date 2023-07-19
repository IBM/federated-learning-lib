import os

from gensim.models.doc2vec import Doc2Vec


def get_hyperparams(model):
    hyperparams = {
        "local": {
            "training": {
                "epochs": 3,
            }
        }
    }
    return hyperparams


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):
    model = Doc2Vec(vector_size=50, min_count=2)
    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    fname = os.path.join(folder_configs, "gensim_doc2vec.model")
    with open(fname, "wb") as file:
        model.save(file)

    spec = {"model_name": "doc2vec", "model_definition": fname}

    model = {
        "name": "Doc2VecFLModel",
        "path": "ibmfl.model.doc2vec_fl_model",
        "spec": spec,
    }

    return model
