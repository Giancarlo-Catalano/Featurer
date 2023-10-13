#!/usr/bin/env python3
import json
import sys

from Version_E.Testing import TestingUtilities


def execute_command_line():
    command_line_arguments = sys.argv
    if len(command_line_arguments) < 2:
       raise Exception("Not enough arguments")

    first_argument = command_line_arguments[1]
    with open(first_argument) as settings_file:
        settings = json.load(settings_file)
        TestingUtilities.run_test(settings)




if __name__ == '__main__':
    execute_command_line()
