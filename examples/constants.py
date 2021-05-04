DEFAULT_CONNECTION = 'default'
DEFAULT_SERVER = 'default'

# Examples helper descriptions
GENERATE_DATA_DESC = "generates data for running IBM FL examples"
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
TASK_NAME_DESC = "task name, specified when using RabbitMQ connection"

EXAMPLES_WARNING = "WARNING:: Usage of -m keras_classifier option  is deprecated and replaced with -m keras -f iter_avg. Ref https://github.com/IBM/federated-learning-lib/blob/main/setup.md for more information"
CONNECTION_TYPE_DESC = "type of connection to use; supported types are flask, rabbitmq and websockets"

# Integration
FL_DATASETS = ["default", "mnist", "nursery", "adult", "federated-clustering", "compas", "german",
               "higgs", "airline", "diabetes", "binovf", "multovf", "linovf", "femnist", "cifar10"]

FL_EXAMPLES = ["iter_avg", "fedavg", "coordinate_median", "gradient_aggregation", "krum", "pfnm", 
                "zeno", "fedprox", "fedplus", "differential_privacy_sgd", "rl_cartpole", 
                "rl_pendulum", "sklearn_logclassification_rw", "spahm", "id3_dt", "prej_remover",
                "sklearn_logclassification_globalrw", "naive_bayes_dp"]

FL_MODELS = ["keras", "pytorch", "tf", "sklearn", "None", "keras_classifier"]

FL_CONN_TYPES = ["flask", "rabbitmq"]
