"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""
#!/usr/bin/env python3

import re
import sys
import logging
import os
from time import sleep

fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)

from ibmfl.party.status_type import StatusType
from ibmfl.party.party import Party

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    """
    Main function can be used to create an application out
    of our Party class which could be interactive
    """
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        logging.error('Please provide yaml configuration')
    config_file = sys.argv[1]
    p = Party(config_file=config_file)

    # Loop over commands passed by runner
    for msg in sys.stdin:
        if re.match('START', msg):
            # Start server
            p.start()

        if re.match('STOP', msg):
            p.connection.stop()
            break

        if re.match('REGISTER', msg):
            p.register_party()

        if re.match('EVAL', msg):
            p.evaluate_model()

    # Stop only when aggregator tells us;
    # in the future, dynamically deciding commands can be supported.
    while p.proto_handler.status != StatusType.STOPPING:
        sleep(1)

    p.stop()
