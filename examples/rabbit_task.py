import argparse

import pycloudmessenger.ffl.abstractions as ffl
import pycloudmessenger.ffl.fflapi as fflapi



def args_parse():
    parser = argparse.ArgumentParser(description='RabbitMQ connection')
    parser.add_argument('--credentials', required=True)
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--password', required=True)
    parser.add_argument('--action', nargs = '?', choices=('create', 'list'), default = 'create')
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
            cmdline.credentials,
            cmdline.user,
            cmdline.password
        )

        user = ffl.Factory.user(context)

        with user:
            if cmdline.action == 'create':
                result = user.create_task(cmdline.task_name, ffl.Topology.star, {})
                print(result)
                print('Task created.')
            elif cmdline.action == 'list':
                tasks = user.get_tasks()
                if tasks:
                    for t in tasks:
                        print(t['task_name'], t['status'])
    except Exception as err:
        print('error: %s', err)
        raise err


if __name__ == '__main__':
    main()
