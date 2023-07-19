#!/usr/bin/env python
# author markpurcell@ie.ibm.com

import argparse
import json

import pycloudmessenger.ffl.abstractions as ffl
import pycloudmessenger.ffl.fflapi as fflapi


def args_parse():
    parser = argparse.ArgumentParser(description="RabbitMQ register")
    parser.add_argument("--credentials", required=True)
    cmdline = parser.parse_args()
    return cmdline


def main():
    cmdline = args_parse()

    ffl.Factory.register("cloud", fflapi.Context, fflapi.User, fflapi.Aggregator, fflapi.Participant)
    context = ffl.Factory.context("cloud", cmdline.credentials)

    print(f"Remove user: {context.user()} ...")

    user = ffl.Factory.user(context)
    with user:
        user.deregister()

    print(f"User successfully removed: {context.user()}")


if __name__ == "__main__":
    main()
