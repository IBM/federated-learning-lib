"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2023 All Rights Reserved.
"""
#!/usr/bin/env python3

import logging
import os
import sys
from time import sleep

from ibmfl.party.party import Party
from ibmfl.party.status_type import StatusType

fl_path = os.path.abspath(".")
if fl_path not in sys.path:
    sys.path.append(fl_path)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    """
    Main function can be used to create an application out \
    of our Aggregator class which could be interactive
    """
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        logger.error("Please provide yaml configuration")
    config_file = sys.argv[1]
    if not os.path.isfile(config_file):
        logger.debug("config file '{}' does not exist".format(config_file))

    """
    Check for end of file marker which indicates copy of the config \
    and data files are completed 
    """
    file_copy_completes = "/tmp/end_of_file_marker.txt"
    while not os.path.isfile(file_copy_completes):
        print("Waiting for config files and datasets")
        sleep(10)

    p = Party(config_file=config_file)
    commands = ["START", "REGISTER"]
    # Loop over commands passed by runner
    for command in commands:
        if command.lower() == ("START").lower():
            # Start server
            p.start()
        if command.lower() == ("STOP").lower():
            p.connection.stop()
            break
        if command.lower() == ("REGISTER").lower():
            p.register_party()

    # Stop only when aggregator tells us;
    # in the future, dynamically deciding commands can be supported.
    while p.proto_handler.status != StatusType.STOPPING:
        sleep(1)

    p.stop()
