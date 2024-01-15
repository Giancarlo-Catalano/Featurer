#!/usr/bin/env python3
import json
import sys
from os import listdir
from os.path import isfile, join
from typing import Callable

import SearchSpace
import utils
from BenchmarkProblems.ArtificialProblem import ArtificialProblem
from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from BenchmarkProblems.PlateauProblem import PlateauProblem
from Version_E.BaselineAlgorithms.GA import GAMiner
from Version_E.BaselineAlgorithms.RandomSearch import random_feature_in_search_space
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.BiDirectionalMiner import BiDirectionalMiner
from Version_E.InterestingAlgorithms.ConstructiveMiner import ConstructiveMiner
from Version_E.InterestingAlgorithms.DestructiveMiner import DestructiveMiner
from Version_E.InterestingAlgorithms.Miner import FeatureSelector, run_until_found_features
from Version_E.InterestingAlgorithms.Miner import run_until_fixed_amount_of_iterations, run_with_limited_budget
from Version_E.MeasurableCriterion.CriterionUtilities import Balance
from Version_E.MeasurableCriterion.Explainability import TrivialExplainability
from Version_E.MeasurableCriterion.GoodFitness import HighFitness
from Version_E.MeasurableCriterion.Interaction import Interaction
from Version_E.MeasurableCriterion.SHAPValue import SHAPValue
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Testing import TestingUtilities, Problems, Criteria, Tests, CSVGenerators
from Version_E.Sampling.GASampler import GASampler
from Version_E.Testing.TestingUtilities import execute_and_time


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
    CSVGenerators.make_csv_for_bgb(files_in_directory, output_name)


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


    criterion = {"which": "balance",
                 "arguments":  [{"which": "simple"},
                                {"which": "high_fitness"},
                                {"which": "interaction"}],
                 "weights": [1, 1, 1]}

    problem = plateau

    problem = Problems.decode_problem(problem)
    criterion = Criteria.decode_criterion(criterion, problem)
    sample_size = 10000
    training_ppi = TestingUtilities.get_evolved_population_sample(problem, sample_size, -1)
    selector = FeatureSelector(training_ppi, criterion)

    miner = ConstructiveMiner(selector=selector,
                                 population_size=150,
                                 stochastic=False,
                                 uses_archive=True,
                                 termination_criteria_met=run_with_limited_budget(budget_limit = 10000))

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
    folder_names = ["bgb"]
    folder_root = r"/home/gian/Documents/CondorDataCollection/PSs/Jan_15"

    for folder_name in folder_names:
        print(f"Aggregating {folder_name}")
        input_full_path = folder_root + "/" + folder_name
        output_full_path = folder_root + "/csvs/" + folder_name + ".csv"
        aggregate_files(input_full_path, output_full_path)


def test_other():
    problem = PlateauProblem(5)
    ga = GASampler(fitness_function = problem.score_of_candidate,
                   search_space = problem.search_space,
                   population_size = 150,
                   termination_criteria = run_with_limited_budget(100000))
    ga.sample(50)


def time_evaluate_fs(amount_full_solutions: int,
                     problem: CombinatorialProblem) -> float:
    fitness_function = problem.score_of_candidate
    random_fss = [problem.get_random_candidate_solution() for _ in range(amount_full_solutions)]
    def evaluate_all():
        fitnesses = [fitness_function(fs) for fs in random_fss]
        return len(fitnesses)  # bogus result

    _, execution_time = execute_and_time(evaluate_all)
    return execution_time

def time_evaluate_ps(amount_partial_solutions: int,
                     problem: CombinatorialProblem) -> float:
    reference_population_size = 10000
    reference_fs = [problem.get_random_candidate_solution()
                    for _ in range(reference_population_size)]
    reference_fitnesses = [problem.score_of_candidate(c) for c in reference_fs]
    reference_ppi = PrecomputedPopulationInformation(problem.search_space, reference_fs, reference_fitnesses)

    criterion = Balance([HighFitness(),
                         TrivialExplainability(),
                         Interaction()], weights=[1, 1, 1])


    random_fs = [random_feature_in_search_space(problem.search_space)
                 for _ in range(amount_partial_solutions)]
    def evaluate_all():
        pfi = PrecomputedFeatureInformation(reference_ppi, random_fs)
        fitnesses = criterion.get_score_array(pfi)
        return len(fitnesses)

    _, execution_time = execute_and_time(evaluate_all)
    return execution_time



def test_time_comparison():
    amount = 10000

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

    problems = [Problems.decode_problem(p) for p in [trapk, plateau, artificial_problem]]
    problem_names = ["artificial", "trapk", "plateau"]

    for repetition in range(10):
        for amount in [100, 1000, 10000]:
            for name, problem in zip(problem_names, problems):
                time_for_fs = time_evaluate_fs(amount, problem)
                time_for_ps = time_evaluate_ps(amount, problem)

                print(f"{name}\t{amount}\t{time_for_fs}\t{time_for_ps}")


if __name__ == '__main__':
    execute_command_line()
    # test_new_miner()
    # aggregate_folders()
    # test_other()


    # test_time_comparison()




