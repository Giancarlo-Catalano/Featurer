#!/usr/bin/env python3
import json
import sys
from os import listdir
from os.path import isfile, join

from Version_E.Testing import TestingUtilities
from Version_E.Testing.Miners import aggregate_algorithm_jsons_into_csv


def execute_command_line():
    command_line_arguments = sys.argv
    if len(command_line_arguments) < 2:
       raise Exception("Not enough arguments")

    first_argument = command_line_arguments[1]
    with open(first_argument) as settings_file:
        settings = json.load(settings_file)
        TestingUtilities.run_test(settings)


def aggregate_files(directory:str, output_name: str, for_time):
    #get files in directory, with absolute path
    files_in_directory = [join(directory, file) for file in listdir(directory)]
    #remove non-files
    files_in_directory = [file for file in files_in_directory if isfile(file)]

    #aggregate
    aggregate_algorithm_jsons_into_csv(files_in_directory, output_name, for_time=for_time)


if __name__ == '__main__':
    #execute_command_line()
    input_directory = "C:\\Users\\gac8\\Documents\\outputs\\Pss\\algo_comparison\\run_6"
    aggregate_files(input_directory, "all_runs_times_smaller_problem.csv", for_time=True)
    aggregate_files(input_directory, "all_runs_successes_smaller_problem.csv", for_time=False)
