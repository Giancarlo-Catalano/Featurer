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
from Version_E.InterestingAlgorithms.Miner import FeatureSelector
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
    CSVGenerators.make_csv_for_limited_budget_run(files_in_directory, output_name)


def get_evolved_population_sample(problem: BenchmarkProblems.CombinatorialProblem.CombinatorialProblem,
                           population_size: int,
                           amount_of_generations: int) -> PrecomputedPopulationInformation:
    ga = GASampler(problem.score_of_candidate, problem.search_space, population_size, amount_of_generations)
    population = ga.evolve_population()
    fitness_list = [problem.score_of_candidate(candidate) for candidate in population]

    return PrecomputedPopulationInformation(problem.search_space, population, fitness_list)


def test_new_miner():
    artificial_problem = {"which": "artificial",
                          "size": 20,
                          "size_of_partials": 4,
                          "amount_of_features": 4,
                          "allow_overlaps": True}

    insular_problem = {"which": "insular",
                       "amount_of_islets": 4}

    trapk = {"which": "trapk",
             "amount_of_groups": 3,
             "k": 5}

    plateau = {"which": "plateau",
               "amount_of_groups": 6}


    problem = trapk

    criterion = {"which": "balance",
                 "arguments": [
                     {"which": "explainability"},
                     {"which": "weakest_link"},
                     {"which": "high_fitness"},
                     {"which": "atomicity"}
                 ],
                 "weights": [1, 1, 1, 1]}

    problem = Problems.decode_problem(problem)
    criterion = Criteria.decode_criterion(criterion, problem)
    sample_size = 1500
    #training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)
    training_ppi = get_evolved_population_sample(problem, sample_size, 1)
    selector = FeatureSelector(training_ppi, criterion)

    biminer = BiDirectionalMiner(selector=selector,
                                 population_size=150,
                                 stochastic=False,
                                 uses_archive=False,
                                 termination_criteria_met=run_with_limited_budget(10000))

    ga_miner = GAMiner(selector=selector,
                       population_size=120,
                       termination_criteria_met=run_with_limited_budget(10000))

    miner = biminer

    print(f"The problem is {problem.long_repr()}")
    print(f"The miner is {miner}")

    good_features = miner.get_meaningful_features(120)
    print("The good features are: ")
    for feature in good_features:
        print(problem.feature_repr(feature.to_legacy_feature()))
        print(criterion.describe_feature(feature, training_ppi))

    evaluations = miner.feature_selector.used_budget
    print(f"The used budget is {evaluations}")

    print("The ideal features have the following scores")
    ideals = problem.get_ideal_features()
    for feature in ideals:
        print(problem.feature_repr(feature.to_legacy_feature()))
        print(criterion.describe_feature(feature, training_ppi))



def aggregate_folders():
    folder_names = ["trap5"]
    folder_root = r"C:\Users\gac8\Documents\R projects\PS_analysis\input_files\Nov-22"

    for folder_name in folder_names:
        print(f"Aggregating {folder_name}")
        input_full_path = folder_root + "\\" + folder_name
        output_full_path = folder_root + "\\csvs\\" + folder_name + ".csv"
        aggregate_files(input_full_path, output_full_path)


if __name__ == '__main__':
    # execute_command_line()
    test_new_miner()
    # aggregate_folders()
