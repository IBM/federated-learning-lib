#!/usr/bin/env python
#author markpurcell@ie.ibm.com

import argparse

import pycloudmessenger.ffl.abstractions as ffl
import pycloudmessenger.ffl.fflapi as fflapi



def args_parse():
    parser = argparse.ArgumentParser(description='RabbitMQ connection')
    parser.add_argument('--credentials', required=True)
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--action', nargs = '?', choices=('create', 'list', 'lineage'), default = 'create')
    cmdline = parser.parse_args()
    return cmdline


def main():
    try:
        cmdline = args_parse()

        ffl.Factory.register(
            'cloud',
            fflapi.Context,
            fflapi.User,
            fflapi.Aggregator,
            fflapi.Participant
        )

        context = ffl.Factory.context(
            'cloud',
            cmdline.credentials
        )

        user = ffl.Factory.user(context)

        with user:
            if cmdline.action == 'create':
                result = user.create_task(cmdline.task_name, ffl.Topology.star, {})
                print(result)
                print('Task created.')
            elif cmdline.action == 'list':
                tasks = user.get_tasks('FAILED')
                if tasks:
                    for t in tasks:
                        print(t['task_name'], t['status'])
            elif cmdline.action == 'lineage':
                result = user.model_lineage(cmdline.task_name)
                print_lineage(result)
    except Exception as err:
        print(f'error: {err}')
        raise err


def print_lineage(result: dict):
    training_round = 0

    print(f"{'Round':5} {'Date':30} {'Origin':20} {'Id':11} {'Hash':11} {'Value Estimate':20} {'Reward':10}")

    for line in result:
        if 'genre' in line:
            if line['genre'] == 'INTERIM':
                training_round += 1
                print(f"{training_round:^5d} {line['added']:30} {'AGGREGATOR':20} " +
                            f"...{str(line['external_id'][-7:]):8} ...{str(line['xsum'][-7:]):8}")
            elif line['genre'] == 'COMPLETE':
                print(f"Final {line['added']:30} {'AGGREGATOR':20} " +
                            f"...{str(line['external_id'][-7:]):8} ...{str(line['xsum'][-7:]):8}")
            else:
                print(f"{training_round:^5d} {line['added']:30} {line['participant']:20} " +
                            f"...{str(line['external_id'][-7:]):8} ...{str(line['xsum'][-7:]):8} " +
                            f"{str(line['contribution']):20} {str(line['reward']):10}")
        else:
            training_round += 1
            print(f"{training_round:^5d} {line['added']:30} {line['metadata']:20} " +
                    f"{str(''):11} ...{str(line['xsum'][-7:]):8} " +
                    f"{str(line['contribution']):20} {str(line['reward']):10}")


if __name__ == '__main__':
    main()
