#!/usr/bin/env python3
import json
import sys
from os import listdir
from os.path import isfile, join

import BenchmarkProblems.CombinatorialProblem
import SearchSpace
from Version_E.BaselineAlgorithms.GA import GAMiner
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.BiDirectionalMiner import BiDirectionalMiner
from Version_E.InterestingAlgorithms.ConstructiveMiner import ConstructiveMiner
from Version_E.InterestingAlgorithms.DestructiveMiner import DestructiveMiner
from Version_E.InterestingAlgorithms.Miner import FeatureSelector, run_until_found_features
from Version_E.InterestingAlgorithms.Miner import run_for_fixed_amount_of_iterations, run_with_limited_budget
from Version_E.MeasurableCriterion.SHAPValue import SHAPValue
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Testing import TestingUtilities, Problems, Criteria, Tests, CSVGenerators
from Version_E.Sampling.GASampler import GASampler


def execute_command_line():
    command_line_arguments = sys.argv
    if len(command_line_arguments) < 2:
        raise Exception("Not enough arguments")

    first_argument = command_line_arguments[1]
    with open(first_argument) as settings_file:
        settings = json.load(settings_file)
        Tests.run_test(settings)


def aggregate_files(directory: str, output_name: str):
    # get files in directory, with absolute path
    files_in_directory = [join(directory, file) for file in listdir(directory)]
    # remove non-files
    files_in_directory = [file for file in files_in_directory if isfile(file)]

    # aggregate
    CSVGenerators.make_csv_for_budget_needed_run(files_in_directory, output_name)





def test_new_miner():
    artificial_problem = {"which": "artificial",
                          "size": 15,
                          "size_of_partials": 4,
                          "amount_of_features": 5,
                          "allow_overlaps": True}

    trapk = {"which": "trapk",
             "amount_of_groups": 5,
             "k": 5}

    plateau = {"which": "plateau",
               "amount_of_groups": 5}

    checkerboard = {"which": "checkerboard",
                    "rows": 4,
                    "cols": 4}


    problem = trapk

    criterion = {"which": "balance",
                 "arguments":  [{"which": "simple"},
                                {"which": "high_fitness"},
                                {"which": "interaction"}],
                 "weights": [1, 1, 1]}

    problem = Problems.decode_problem(problem)
    criterion = Criteria.decode_criterion(criterion, problem)
    sample_size = 10000
    training_ppi = TestingUtilities.get_evolved_population_sample(problem, sample_size, -1)
    selector = FeatureSelector(training_ppi, criterion)

    miner = ConstructiveMiner(selector=selector,
                                 population_size=100,
                                 stochastic=False,
                                 uses_archive=True,
                                 termination_criteria_met=run_with_limited_budget(budget_limit = 20000))

    print(f"The problem is {problem}")
    print(f"It has ideals \n\t" +"\n\t".join(f"{ideal}" for ideal in problem.get_ideal_features()))

    print(f"The miner is {miner}")


    good_features = miner.get_meaningful_features(30)
    print("The good features are: ")
    for feature in good_features:
        print(problem.feature_repr(feature.to_legacy_feature()))
        print(criterion.describe_feature(feature, training_ppi))

    evaluations = miner.feature_selector.used_budget
    print(f"The used budget is {evaluations}")

    for feature in problem.get_ideal_features():
        print(problem.feature_repr(feature.to_legacy_feature()))
        print(criterion.describe_feature(feature, training_ppi))



def aggregate_folders():
    folder_names = ["mixed"]
    folder_root = r"C:\Users\gac8\Documents\R projects\PS_analysis\input_files\Dec-19"

    for folder_name in folder_names:
        print(f"Aggregating {folder_name}")
        input_full_path = folder_root + "\\" + folder_name
        output_full_path = folder_root + "\\csvs\\" + folder_name + ".csv"
        aggregate_files(input_full_path, output_full_path)


if __name__ == '__main__':
    #execute_command_line()
    test_new_miner()
    #aggregate_folders()
