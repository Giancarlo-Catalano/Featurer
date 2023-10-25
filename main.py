#!/usr/bin/env python3
import json
import sys
from os import listdir
from os.path import isfile, join

from Version_E.InterestingAlgorithms.GCMiner import run_for_fixed_amount_of_iterations
from Version_E.InterestingAlgorithms.BiDirectionalMiner import BiDirectionalMiner
from Version_E.InterestingAlgorithms.ConstructiveMiner import ConstructiveMiner
from Version_E.InterestingAlgorithms.DestructiveMiner import DestructiveMiner
from Version_E.InterestingAlgorithms.Miner import FeatureSelector
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.SimpleSampler import SimpleSampler
from Version_E.Testing import TestingUtilities, Problems, Criteria
from Version_E.Testing.Miners import aggregate_algorithm_jsons_into_csv


def execute_command_line():
    command_line_arguments = sys.argv
    if len(command_line_arguments) < 2:
        raise Exception("Not enough arguments")

    first_argument = command_line_arguments[1]
    with open(first_argument) as settings_file:
        settings = json.load(settings_file)
        TestingUtilities.run_test(settings)


def aggregate_files(directory: str, output_name: str, for_time):
    # get files in directory, with absolute path
    files_in_directory = [join(directory, file) for file in listdir(directory)]
    # remove non-files
    files_in_directory = [file for file in files_in_directory if isfile(file)]

    # aggregate
    aggregate_algorithm_jsons_into_csv(files_in_directory, output_name, for_time=for_time)


def test_new_miner():
    artificial_problem = {"which": "artificial",
                          "size": 25,
                          "size_of_partials": 5,
                          "amount_of_features": 5,
                          "allow_overlaps": False}

    checkerboard_problem = {"which": "checkerboard",
                            "rows": 8,
                            "cols": 8}


    trapk = {"which": "trapk",
             "amount_of_groups": 3,
             "k": 5}

    problem = checkerboard_problem

    criterion = {"which": "balance",
                 "arguments": [
                     {"which": "high_fitness"},
                     {"which": "consistent_fitness"},
                     {"which": "explainability"}
                 ],
                 "weights": [1, 1, 1]}

    problem = Problems.decode_problem(problem)
    criterion = Criteria.decode_criterion(criterion, problem)
    sample_size = 2400
    training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)
    selector = FeatureSelector(training_ppi, criterion)

    miner = BiDirectionalMiner(selector=selector,
                               population_size=60,
                               stochastic=True,
                               uses_archive=True,
                               termination_criteria_met=run_for_fixed_amount_of_iterations(30))

    print(f"The problem is {problem.long_repr()}")
    print(f"The miner is {miner}")

    good_features = miner.get_meaningful_features(60)
    print("The good features are: ")
    for feature in good_features:
        print(problem.feature_repr(feature.to_legacy_feature()))
        print(criterion.describe_feature(feature, training_ppi))

    evaluations = miner.feature_selector.used_budget
    print(f"The used budget is {evaluations}")


    scored_good_features = miner.with_scores(good_features)
    sampler = SimpleSampler(problem.search_space, scored_good_features)


    for _ in range(12):
        sampled_candidate = sampler.sample_candidate()
        fitness = problem.score_of_candidate(sampled_candidate)
        print(problem.candidate_repr(sampled_candidate))
        print(f"And the fitness is {fitness}\n")


if __name__ == '__main__':
    # execute_command_line()
    # input_directory = "C:\\Users\\gac8\\Documents\\outputs\\Pss\\algo_comparison\\run_7"
    # aggregate_files(input_directory, "all_runs_times_smaller_problem.csv", for_time=True)
    # aggregate_files(input_directory, "all_runs_successes_smaller_problem.csv", for_time=False)

    test_new_miner()
