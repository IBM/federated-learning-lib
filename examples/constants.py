DEFAULT_CONNECTION = 'default'
DEFAULT_SERVER = 'default'

# Examples helper descriptions
GENERATE_DATA_DESC = "generates data for running FL examples"
NUM_PARTIES_DESC = "the number of parties to split the data into"
DATASET_DESC = "which data set to use"
PATH_DESC = "directory to save the data"
PER_PARTY = "the number of data points per party"
STRATIFY_DESC = "proportionally stratify the data according to the source distribution"
CONF_PATH = "directory to save the configs"

NEW_DESC = "create a new directory for this run based on current time instead of overriding"
NAME_DESC = "the name of the run (default is current time)"
PER_PARTY_ERR = "points per party must either specify one number of a list equal to num_parties"
GENERATE_CONFIG_DESC = "generates aggregator and party configuration files"
PATH_CONFIG_DESC = "path to load saved config data"
FUSION_CONFIG_DESC = "which fusion example to run"
MODEL_CONFIG_DESC = "which model to use for fusion example"
TASK_NAME_DESC = "task name, specified when using PubSub connection"

CONNECTION_TYPE_DESC = "type of connection to use; supported types are flask and pubsub"

CONTEXT_PATH = "context directory to import the generate script from different folders other that examples"

# Integration
FL_DATASETS = ["default", "mnist", "nursery", "adult", "federated-clustering", "compas", "german",
               "higgs", "airline", "diabetes", "binovf", "multovf", "linovf", "femnist", "cifar10", "custom_dataset"]
               
FL_EXAMPLES = ["iter_avg", "fedavg", "coordinate_median", "gradient_aggregation", "krum", "pfnm", 
                "zeno", "fedprox", "fedavgplus", 
                "differential_privacy_sgd", 
                "rl_cartpole", "rl_pendulum", "sklearn_logclassification_rw", "spahm",
                "sklearn_logclassification_globalrw", "naive_bayes_dp", "id3_dt", "prej_remover", "iter_avg_openshift", "shuffle_iter_avg",
                "coordinate_median_plus", "geometric_median_plus", "doc2vec", "comparative_elimination"]
FL_MODELS = ["keras", "pytorch", "tf", "sklearn", "doc2vec", "None"]

FL_CONN_TYPES = ["flask", "pubsub"]

FL_CONTEXT = {'openshift':'openshift_fl.examples'}
