#!/usr/bin/env python3
import json
import sys
from os import listdir
from os.path import isfile, join

from Version_E.BaselineAlgorithms.GA import GAMiner
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.BiDirectionalMiner import BiDirectionalMiner
from Version_E.InterestingAlgorithms.Miner import FeatureSelector
from Version_E.InterestingAlgorithms.Miner import run_for_fixed_amount_of_iterations, run_with_limited_budget
from Version_E.MeasurableCriterion.SHAPValue import SHAPValue
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Testing import TestingUtilities, Problems, Criteria, Tests


def execute_command_line():
    command_line_arguments = sys.argv
    if len(command_line_arguments) < 2:
        raise Exception("Not enough arguments")

    first_argument = command_line_arguments[1]
    with open(first_argument) as settings_file:
        settings = json.load(settings_file)
        Tests.run_test(settings)


def aggregate_files(directory: str, output_name: str, for_time):
    # get files in directory, with absolute path
    files_in_directory = [join(directory, file) for file in listdir(directory)]
    # remove non-files
    files_in_directory = [file for file in files_in_directory if isfile(file)]

    # aggregate
    #aggregate_algorithm_jsons_into_csv(files_in_directory, output_name, for_time=for_time)


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
                 "weights": [3, 1, 1]}

    problem = Problems.decode_problem(problem)
    criterion = Criteria.decode_criterion(criterion, problem)
    sample_size = 2400
    training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)
    selector = FeatureSelector(training_ppi, criterion)

    biminer = BiDirectionalMiner(selector=selector,
                               population_size=60,
                               stochastic=True,
                               uses_archive=True,
                               termination_criteria_met=run_for_fixed_amount_of_iterations(30))

    ga_miner = GAMiner(selector = selector,
                       population_size = 60,
                       termination_criteria_met=run_with_limited_budget(100 ** 2))

    miner = biminer

    print(f"The problem is {problem.long_repr()}")
    print(f"The miner is {miner}")

    good_features = miner.get_meaningful_features(60)
    print("The good features are: ")
    for feature in good_features:
        print(problem.feature_repr(feature.to_legacy_feature()))
        print(criterion.describe_feature(feature, training_ppi))

    evaluations = miner.feature_selector.used_budget
    print(f"The used budget is {evaluations}")

    # scored_good_features = miner.with_scores(good_features)
    # sampler = SimpleSampler(problem.search_space, scored_good_features)
    #
    # print("Running the traditional sampler yields")
    # for _ in range(12):
    #     sampled_candidate = sampler.sample_candidate()
    #     fitness = problem.score_of_candidate(sampled_candidate)
    #     print(problem.candidate_repr(sampled_candidate))
    #     print(f"And the fitness is {fitness}\n")
    #
    # pfi = PrecomputedFeatureInformation(training_ppi, good_features)
    # sampling_sub_criterion = Balance([HighFitness(), ConsistentFitness()])
    # sampling_criteria = All([Completeness(),
    #                          ExpectedFitness(criterion = sampling_sub_criterion, pfi=pfi)])
    # sampling_selector = FeatureSelector(training_ppi, sampling_criteria)
    # sampling_miner = BiDirectionalMiner(sampling_selector,
    #                                     population_size=60,
    #                                     stochastic=True,
    #                                     uses_archive=True,
    #                                     termination_criteria_met=run_for_fixed_amount_of_iterations(30))
    #
    # sampled_from_miner = sampling_miner.get_meaningful_features(12)
    # sampled_from_miner = [feature.to_candidate() for feature in sampled_from_miner]
    # print("Running the ouroboros sampler yields")
    # for feature_sampled_from_miner in sampled_from_miner:
    #     fitness = problem.score_of_candidate(feature_sampled_from_miner)
    #     print(problem.candidate_repr(feature_sampled_from_miner))
    #     print(f"And the fitness is {fitness}\n")



def test_new_criterion():
    artificial_problem = {"which": "artificial",
                          "size": 25,
                          "size_of_partials": 5,
                          "amount_of_features": 5,
                          "allow_overlaps": False}

    checkerboard_problem = {"which": "checkerboard",
                            "rows": 4,
                            "cols": 4}

    trapk = {"which": "trapk",
             "amount_of_groups": 3,
             "k": 5}

    problem = artificial_problem

    problem = Problems.decode_problem(problem)
    criterion = SHAPValue(sample_size = 600)
    sample_size = 2400
    training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)
    selector = FeatureSelector(training_ppi, criterion)

    print("Generated the selector and the criterion successfully")
    def feature_with_set_zero_at(index: int) -> Feature:
        empty_feature = Feature.empty_feature(problem.search_space)
        return empty_feature.with_value(var_index=index, val = 0)

    features_to_assess = [feature_with_set_zero_at(index) for index in range(problem.search_space.dimensions)]

    scores = selector.get_scores(features_to_assess)
    for feature, score in zip(features_to_assess, scores):
        print(f"{feature}\n{score}")
        print(criterion.describe_feature(feature, training_ppi))




if __name__ == '__main__':
    #execute_command_line()
    # input_directory = "C:\\Users\\gac8\\Documents\\outputs\\Pss\\algo_comparison\\run_7"
    # aggregate_files(input_directory, "all_runs_times_smaller_problem.csv", for_time=True)
    # aggregate_files(input_directory, "all_runs_successes_smaller_problem.csv", for_time=False)

    #test_new_miner()

    test_new_criterion()
