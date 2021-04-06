#!/usr/bin/env python3

import argparse
import os
import time
import yaml
import json
import sys
from importlib import import_module

import pycloudmessenger.ffl.abstractions as ffl
import pycloudmessenger.ffl.fflapi as fflapi


fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)

from examples.constants import GENERATE_CONFIG_DESC, NUM_PARTIES_DESC, \
    PATH_CONFIG_DESC, MODEL_CONFIG_DESC, NEW_DESC, NAME_DESC, \
    FL_EXAMPLES, FL_CONN_TYPES, CONNECTION_TYPE_DESC, TASK_NAME_DESC


def check_valid_folder_structure(p):
    """
    Checks that the folder structure is valid

    :param p: an argument parser
    :type p: argparse.ArgumentParser
    """
    for folder in FL_EXAMPLES:
        if not os.path.isfile(os.path.join("examples", folder, "README.md")) and not os.path.isfile(os.path.join(
                "examples", folder, "generate_configs.py")):
            p.error(
                "Bad folder structure: '{}' directory is missing files.".format(folder))


def setup_parser():
    """
    Sets up the parser for Python script

    :return: a command line parser
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(description=GENERATE_CONFIG_DESC)
    p.add_argument("--num_parties", "-n", help=NUM_PARTIES_DESC,
                   type=int, required=True)
    p.add_argument("--dataset", "-d",
                   help="Dataset code from examples", type=str, required=True)

    p.add_argument("--data_path", "-p", help=PATH_CONFIG_DESC, required=True)
    p.add_argument("--model", "-m", choices=[os.path.basename(d) for d in FL_EXAMPLES],
                   help=MODEL_CONFIG_DESC, required=True)
    p.add_argument("--create_new", "-new", action="store_true", help=NEW_DESC)
    p.add_argument("--name", help=NAME_DESC)
    p.add_argument("--connection", "-c", choices=[os.path.basename(
        d) for d in FL_CONN_TYPES], help=CONNECTION_TYPE_DESC, required=False, default="flask")
    p.add_argument("--task_name", "-t", help=TASK_NAME_DESC, required=False)
    return p


def rabbit_task(credentials: str, aggregator: str, password: str, task_name: str):
    try:
        ffl.Factory.register(
            'cloud',
            fflapi.Context,
            fflapi.User,
            fflapi.Aggregator,
            fflapi.Participant
        )

        context = ffl.Factory.context(
            'cloud',
            credentials,
            aggregator,
            password
        )

        user = ffl.Factory.user(context)

        with user:
            result = user.create_task(task_name, ffl.Topology.star, {})
            print(f"Task '{task_name}' created.")
    except Exception as err:
        print('error: %s', err)
        raise

def generate_connection_config(conn_type, party_id=0, is_party=False, task_name = None):
    connection = {}

    if conn_type == 'flask':
        tls_config = {
            'enable': False
        }
        connection = {
            'name': 'FlaskConnection',
            'path': 'ibmfl.connection.flask_connection',
            'sync': False
        }
        if is_party:
            connection['info'] = {
                'ip': '127.0.0.1',
                'port': 8085 + party_id
            }
        else:
            connection['info'] = {
                'ip': '127.0.0.1',
                'port': 5000
            }
        connection['info']['tls_config'] = tls_config

    if conn_type == 'rabbitmq':
        credentials = os.environ.get('IBMFL_BROKER')
        if not credentials:
            raise Exception("IBMFL_BROKER: environment variable not available.")

        credentials = yaml.load(credentials)
        if 'rabbit' in credentials:
            with open('ibmfl_broker_connection.json', 'w') as creds:
                creds.write(json.dumps(credentials['rabbit']))

        connection = {
            'name': 'RabbitMQConnection',
            'path': 'ibmfl.connection.rabbitmq_connection',
            'sync': True
        }

        if is_party:
            party = credentials[f'party{party_id}']['name']
            password = credentials[f'party{party_id}']['password']

            connection['info'] = {
                'credentials': 'ibmfl_broker_connection.json',
                'user': party,
                'password': password,
                'role': 'party',
                'task_name': task_name
            }
        else:
            aggregator = credentials['aggregator']['name']
            password = credentials['aggregator']['password']
            connection['info'] = {
                'credentials': 'ibmfl_broker_connection.json',
                'user': aggregator,
                'password': password,
                'role': 'aggregator',
                'task_name': task_name
            }

            rabbit_task('ibmfl_broker_connection.json', aggregator, password, task_name)

    return connection


def get_aggregator_info(conn_type):

    if conn_type == 'flask':
        aggregator = {
            'ip': '127.0.0.1',
            'port': 5000
        }
    else:
        aggregator = {}

    return aggregator


def generate_ph_config(conn_type, is_party=False):

    if is_party:
        protocol_handler = {
            'name': 'PartyProtocolHandler',
            'path': 'ibmfl.party.party_protocol_handler'
        }
    else:
        protocol_handler = {
            'name': 'ProtoHandler',
            'path': 'ibmfl.aggregator.protohandler.proto_handler'
        }

    if conn_type == 'rabbitmq':
        protocol_handler['name'] += 'RabbitMQ'

    return protocol_handler


def generate_fusion_config(module):
    gen_fusion_config = getattr(module, 'get_fusion_config')
    
    return gen_fusion_config()
    

def generate_hp_config(module, num_parties):
    gen_hp_config = getattr(module, 'get_hyperparams')
    hp = gen_hp_config()
    hp['global']['parties'] = num_parties

    return hp


def generate_model_config(module, folder_configs, dataset, is_agg=False, party_id=0):
    get_model_config = getattr(module, 'get_model_config')
    model = get_model_config(folder_configs, dataset, is_agg, party_id)

    return model


def generate_lt_config(module,  keys, party_id=None):
    get_local_training_config = getattr(module, 'get_local_training_config')
    lt = get_local_training_config()

    return lt


def generate_datahandler_config(module, party_id, dataset, folder_data, is_agg=False):

    get_data_handler_config = getattr(module, 'get_data_handler_config')
    dh = get_data_handler_config(party_id, dataset, folder_data, is_agg)

    return dh


def generate_agg_config(module, num_parties, conn_type, dataset, folder_data, folder_configs, keys, task_name = None):

    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)
    config_file = os.path.join(folder_configs, 'config_agg.yml')

    content = {
        'connection': generate_connection_config(conn_type, task_name=task_name),
        'fusion': generate_fusion_config(module),
        'hyperparams': generate_hp_config(module, num_parties),
        'protocol_handler': generate_ph_config(conn_type)
    }

    model = generate_model_config(module, folder_configs, dataset, True)
    data = generate_datahandler_config(
        module, 0, dataset, folder_data, True)
    if model:
        content['model'] = model
    if data:
        content['data'] = data
    with open(config_file, 'w') as outfile:
        yaml.dump(content, outfile)

    print('Finished generating config file for aggregator. Files can be found in: ',
          os.path.abspath(os.path.join(folder_configs, 'config_agg.yml')))


def generate_party_config(module, num_parties, conn_type, dataset, folder_data, folder_configs, keys, task_name = None):

    for i in range(num_parties):
        config_file = os.path.join(
            folder_configs, 'config_party' + str(i) + '.yml')

        
        lh = generate_lt_config(module, None, party_id=i)
        

        content = {
            'connection': generate_connection_config(conn_type, i, True, task_name=task_name),
            'data': generate_datahandler_config(module, i, dataset, folder_data),
            'model': generate_model_config(module, folder_configs, dataset, party_id=i),
            'protocol_handler': generate_ph_config(conn_type, True),
            'local_training': lh,
            'aggregator': get_aggregator_info(conn_type)
        }

        with open(config_file, 'w') as outfile:
            yaml.dump(content, outfile)

    print('Finished generating config file for parties. Files can be found in: ',
          os.path.abspath(os.path.join(folder_configs, 'config_party*.yml')))


if __name__ == '__main__':
    # Parse command line options
    parser = setup_parser()
    args = parser.parse_args()
    check_valid_folder_structure(parser)

    # Collect arguments
    num_parties = args.num_parties
    dataset = args.dataset
    party_data_path = args.data_path
    model = args.model
    create_new = args.create_new
    exp_name = args.name
    conn_type = args.connection
    task_name = args.task_name

    # Create folder to save configs
    folder_configs = os.path.join("examples", "configs")

    if create_new:
        folder_configs = os.path.join(
            folder_configs, exp_name if exp_name else str(int(time.time())))
    else:
        folder_configs = os.path.join(folder_configs, model)

    # Import and run generate_configs.py
    config_model = import_module('examples.{}.generate_configs'.format(model))

    


    generate_agg_config(config_model, num_parties, conn_type,
                        dataset, party_data_path, folder_configs,
                        None, task_name)
    generate_party_config(config_model, num_parties, conn_type,
                          dataset, party_data_path, folder_configs,
                          None, task_name)

