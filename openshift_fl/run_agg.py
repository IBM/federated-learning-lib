"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""
#!/usr/bin/env python3

import logging
import os
import sys

from time import sleep

from ibmfl.aggregator.aggregator import Aggregator
from ibmfl.aggregator.states import States
from ibmfl.util.config import get_config_from_file

fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    """
    Main function can be used to create an application out \
    of our Aggregator class which could be interactive
    """
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        logger.error('Please provide yaml configuration')

    server_process = None
    config_file = sys.argv[1]
    if not os.path.isfile(config_file):
        logger.debug("config file '{}' does not exist".format(config_file))

    """
    Check for end of file marker which indicates copy of the config \
    and data files are completed 
    """
    file_copy_marker = '/tmp/end_of_file_marker.txt'
    while not os.path.isfile(file_copy_marker):
        logger.error("Waiting for config files and datasets")
        sleep(10)

    # Read commands from config file
    commands = ['START', 'TRAIN', 'EVAL', 'STOP']
    if os.path.isfile('/tmp/commands.txt'):
        with open('/tmp/commands.txt') as cmds:
            cmd_str = cmds.read()
            commands = cmd_str.split('[', 1)[1].split(']')[0].split(',')

    config_dict = get_config_from_file(config_file)
    n_parties = config_dict['hyperparams']['global']['num_parties']
    logger.info("Going to wait for {} parties to register.".format(n_parties))
    agg = Aggregator(config_file=config_file)
    for command in commands:
        if command.strip().lower() == ('START').lower():
            agg.proto_handler.state = States.CLI_WAIT
            logger.info("State: " + str(agg.proto_handler.state))
            # Start server
            agg.start()
            while agg.proto_handler.get_n_parties() < n_parties:
                sleep(1)
            logger.info("All parties registered!")
            sleep(10)
        elif command.strip().lower() == ('STOP').lower():
            logger.info("Aggregator stop successful")
            while True:
                logger.info("Waiting : Aggregator stop successful")
                sleep(10)
            #agg.stop()
            break
        elif command.strip().lower() == ('TRAIN').lower():
            logger.info("State: " + str(agg.proto_handler.state))
            agg.start_training()
        elif command.strip().lower() == ('SAVE').lower():
            logger.info("State: " + str(agg.proto_handler.state))
            agg.save_model()
        elif command.strip().lower() == ('EVAL').lower():
            logger.info("State: " + str(agg.proto_handler.state))
            agg.eval_model()
        elif command.strip().lower() == ('SYNC').lower():
            logger.info("State: " + str(agg.proto_handler.state))
            agg.model_synch()
