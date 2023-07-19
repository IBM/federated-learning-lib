"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2023 All Rights Reserved.
"""
#!/usr/bin/env python3

import logging
import os
import re
import sys
from time import sleep

fl_path = os.path.abspath(".")
if fl_path not in sys.path:
    sys.path.append(fl_path)

from ibmfl.aggregator.aggregator import Aggregator
from ibmfl.aggregator.states import States
from ibmfl.util.config import get_config_from_file

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """
    Main function can be used to create an application out
    of our Aggregator class which could be interactive
    """
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        logging.error("Please provide yaml configuration")

    server_process = None
    config_file = sys.argv[1]
    config_dict = get_config_from_file(config_file)
    n_parties = config_dict["hyperparams"]["global"]["num_parties"]
    logging.info("Going to wait for {} parties to register.".format(n_parties))

    if not os.path.isfile(config_file):
        logging.error("config file '{}' does not exist".format(config_file))

    agg = Aggregator(config_file=config_file)
    for line in sys.stdin:
        msg = line.strip().upper()

        if re.match("START", msg):
            agg.proto_handler.state = States.CLI_WAIT
            logging.info("State: " + str(agg.proto_handler.state))
            # Start server
            agg.start()
            while agg.proto_handler.get_n_parties() < n_parties:
                sleep(1)
            logging.info("All parties registered!")
            sleep(10)

        elif re.match("STOP", msg):
            agg.proto_handler.state = States.STOP
            logging.info("State: " + str(agg.proto_handler.state))
            agg.stop()
            break

        elif re.match("TRAIN", msg):
            agg.proto_handler.state = States.TRAIN
            logging.info("State: " + str(agg.proto_handler.state))
            success = agg.start_training()
            if not success:
                agg.stop()
                break

        elif re.match("SAVE", msg):
            agg.proto_handler.state = States.SAVE
            logging.info("State: " + str(agg.proto_handler.state))
            agg.save_model()

        elif re.match("EVAL", msg):
            agg.proto_handler.state = States.EVAL
            logging.info("State: " + str(agg.proto_handler.state))
            agg.eval_model()

        elif re.match("SYNC", msg):
            agg.proto_handler.state = States.SYNC
            logging.info("State: " + str(agg.proto_handler.state))
            agg.model_synch()
