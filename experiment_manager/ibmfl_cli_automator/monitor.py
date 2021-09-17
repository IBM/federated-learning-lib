"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""
#!/usr/bin/env python3

import argparse
import subprocess as sp
import select
import sys
import time
import yaml


if __name__ == '__main__':
    """
    We can daemonize our connections to our remote machines, list the FL processes on remote
    machines, or kill FL processes on remote machines. We can either pass a specfic run's metadata
    file, or we can use a 'global' metadata file to list all processes on a list of machines.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['daemonize', 'list', 'kill'])
    parser.add_argument('--config')

    # read metadata config for the specified run
    args = parser.parse_args()
    if (args.config):
        with open(args.config) as config_file:
            config = yaml.load(config_file.read(), Loader=yaml.Loader)
        if 'timestamp' in config:
            machines = [config['agg_machine']] + config['party_machines']
            usernames = [config['agg_username']] + config['party_usernames']
            run_id = config['timestamp'] if 'timestamp' in config else ''
        else:
            machines = config['machines']
            usernames = config['usernames']
            run_id = ''

    localp = sp.Popen('mkdir -p {}/.ssh'.format(config['local_staging_dir']).split())
    exit_code = localp.wait()

    # decide what to run based on input
    if args.action == 'daemonize':
        daemonize_cmd = 'ssh '\
            '-o "ControlMaster=auto" '\
            '-o "ControlPath={}/.ssh/master-%r@%h:%p" '\
            '-o "ControlPersist=yes" '\
            '-Nn {}@{}'
        cmds = [daemonize_cmd.format(config['local_staging_dir'], u, m) for m, u in zip(machines,usernames)]
    elif args.action == 'list':
        if 'timestamp' in config:
            list_cmd = \
                'ssh -o "ControlMaster=no" -o "ControlPath={}/.ssh/master-%r@%h:%p" {}@{} '\
                '"set -o pipefail; '\
                'pgrep -u {} -f \\"bash.*run_agg\.py.*{}|bash.*run_party\.py.*{}\\" '\
                '| xargs --no-run-if-empty -I@ pgrep -P @ -f \\"run\\" -a"'
            cmds = [list_cmd.format(config['local_staging_dir'], u, m, u, run_id, run_id, u, run_id, run_id) for m, u in zip(machines, usernames)]
        else:
            list_cmd = \
                'ssh -o "ControlMaster=no" -o "ControlPath={}/.ssh/master-%r@%h:%p" {}@{} ' \
                '"set -o pipefail; '\
                'pgrep -f \\"bash.*run_agg\.py|bash.*run_party\.py\\" '\
                '| tee >(xargs --no-run-if-empty -I@ pgrep -P @) '\
                '| xargs --no-run-if-empty ps -o user:8,pid,ppid,cmd p"'
            cmds = [list_cmd.format(config['local_staging_dir'], u, m) for m, u in zip(machines, usernames)]
    elif args.action == 'kill':
        if 'timestamp' in config:
            kill_cmd = \
                'ssh -o "ControlMaster=no" -o "ControlPath={}/.ssh/master-%r@%h:%p" {}@{} '\
                '"set -o pipefail; '\
                'pgrep -u {} -f \\"bash.*run_agg\.py.*{}|run_party\.py.*{}\\" '\
                '| tee >(xargs --no-run-if-empty pgrep -P) | tee >(xargs --no-run-if-empty kill)"'
            cmds = [kill_cmd.format(config['local_staging_dir'], u, m, u, run_id, run_id, u, run_id, run_id) for m, u in zip(machines, usernames)]
        else:
            kill_cmd = \
                'ssh -o "ControlMaster=no" -o "ControlPath={}/.ssh/master-%r@%h:%p" {}@{} '\
                '"set -o pipefail; '\
                'pgrep -u {} -f \\"run_agg\.py|run_party\.py\\" '\
                '&& pkill -u {} -f \\"run_agg\.py|run_party\.py\\""'
            cmds = [kill_cmd.format(config['local_staging_dir'], u, m, u, u) for m, u in zip(machines, usernames)]
    else:
        print('Action not handled. Exiting.')
        exit(1)

    # start all processes
    procs = [sp.Popen(c, stdout=sp.PIPE, stderr=sp.PIPE, shell=True, universal_newlines=True) for c in cmds]
    stdout = ['' for _ in machines]
    stderr = ['' for _ in machines]
    loops = 0

    # wait for output and finally exit when processes end, obtaining all output
    polls = list(p.poll() for p in procs)
    while any(r == None for r in polls):
        ret = select.select([p.stdout.fileno() for p,r in zip(procs,polls) if r == None], [], [])
        for fd in ret[0]:
            for i,p in enumerate(procs):
                if p.stdout.fileno() == fd:
                    stdout[i] += '\t{}'.format(p.stdout.readline())
        polls = tuple(p.poll() for p in procs)
        loops += 1
    for i,p in enumerate(procs):
        for line in p.stdout:
            stdout[i] += '\t{}'.format(line)
        for line in p.stderr:
            stderr[i] += '\t{}'.format(line)
        if not stdout[i].strip():
            stderr[i] += '\tNo processes found.\n'

    # print output
    if args.action != 'daemonize':
        for i,m in enumerate(machines):
            print("{}:".format(m))
            if stdout[i].strip():
                print(stdout[i])
            if stderr[i].strip():
                print(stderr[i])

