#!/usr/bin/env python
#author markpurcell@ie.ibm.com

import json
import argparse
import pycloudmessenger.ffl.fflapi as fflapi
import pycloudmessenger.ffl.abstractions as ffl

def args_parse():
    parser = argparse.ArgumentParser(description='RabbitMQ register')
    parser.add_argument('--credentials', required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--password', required=True)
    parser.add_argument('--org', required=False, default='IBM', help='User organisation')
    cmdline = parser.parse_args()
    return cmdline


def main():
    cmdline = args_parse()

    creds = fflapi.create_user(cmdline.user, cmdline.password, cmdline.org, cmdline.credentials)
    print(json.dumps(creds['connection'], indent=4))


if __name__ == '__main__':
    main()
