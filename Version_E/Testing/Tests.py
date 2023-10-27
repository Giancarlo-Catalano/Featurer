import itertools
import json
import math
import time
import random
from typing import Callable, Iterable

import numpy as np

from BenchmarkProblems.GraphColouring import GraphColouringProblem
import utils
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem, CombinatorialProblem
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, FeatureSelector, run_with_limited_budget, \
    run_until_found_features
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Testing import Problems, Criteria, Miners, TestingUtilities
from collections import defaultdict

from Version_E.Testing.TestingUtilities import execute_and_time

Settings = dict
TestResults = dict

TerminationPredicate = Callable

# order of operations:
"""
    problem = decode_problem(arguments)
    criterion = decode_criterion(arguments, problem)
    termination_predicate = decode_termination_predicate(arguments, problem)
    
    sample_size = decode_sample_size(arguments)
    selector = make_selector(problem, sample_size)
    miner = decode_miner(arguments, selector, termination_predicate)


"""


def decode_termination_predicate(problem: CombinatorialProblem, test_settings: Settings) -> TerminationPredicate:
    test_kind = test_settings["which"]

    if test_kind == "run_with_limited_budget":
        budget: int = test_settings["budget"]
        return run_with_limited_budget(budget)
    elif test_kind == "run_until_success":
        budget: int = test_settings["budget"]
        problem: TestableCombinatorialProblem = problem  # we assume that it is a TestableCombinatorialProblem
        target_individuals = problem.get_ideal_features()
        return run_until_found_features(target_individuals, max_budget=budget)
    else:
        raise Exception(f"Could not generate a termination function for the following test settings: {test_settings}")


def decode_problem(arguments: Settings) -> CombinatorialProblem:
    return Problems.decode_problem(arguments["problem"])


def decode_criterion(arguments: Settings, problem: CombinatorialProblem) -> MeasurableCriterion:
    return Criteria.decode_criterion(arguments["criterion"], problem)


def decode_sample_size(arguments: Settings) -> int:
    return arguments["test"]["sample_size"]


def make_selector(problem: CombinatorialProblem, sample_size: int, criterion: MeasurableCriterion) -> FeatureSelector:
    training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)
    selector = FeatureSelector(training_ppi, criterion)
    return selector


def decode_miner(miner_arguments: Settings, selector, termination_predicate: TerminationPredicate) -> FeatureMiner:
    return Miners.decode_miner(miner_arguments, selector, termination_predicate)


def generate_problem_miner(arguments: Settings, overloading_miner_arguments=None) -> (
        CombinatorialProblem, FeatureMiner):
    problem = Problems.decode_problem(arguments["problem"])
    criterion = Criteria.decode_criterion(arguments["criterion"], problem)
    sample_size = arguments["test"]["sample_size"]
    training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)
    selector = FeatureSelector(training_ppi, criterion)

    if overloading_miner_arguments is None:
        overloading_miner_arguments = arguments["miner"]
    miner = Miners.decode_miner(overloading_miner_arguments, selector)
    return problem, miner



def test_get_distribution(arguments: Settings, runs: int,
                          features_per_run: int) -> TestResults:
    def register_feature(feature: Feature, accumulator):
        mask_array = np.array(feature.variable_mask.tolist())
        accumulator += mask_array

    def single_run():
        problem, miner = generate_problem_miner(arguments)
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, features_per_run)
        cell_coverings = np.zeros(problem.search_space.dimensions, dtype=int)
        for feature in mined_features:
            register_feature(feature, cell_coverings)
        return {"position_counts": [int(count) for count in cell_coverings],
                "runtime": execution_time}

    result = {"test_results": [single_run() for _ in range(runs)]}

    def get_flat_distribution(results: TestResults) -> np.ndarray:
        counts = np.array([item["position_counts"] for item in results["test_results"]])
        counts = np.sum(counts, axis=0)
        return counts

    def print_row_distribution(results: TestResults):
        print("\t".join([f"{item}" for item in get_flat_distribution(results)]))

    def print_table_distribution(results: TestResults):
        counts = get_flat_distribution(results)
        rows = math.floor(math.sqrt(len(counts)))  # assumes it's a square
        counts = counts.reshape((rows, -1))
        for row in counts:
            print("\t".join([f"{item}" for item in row]))

    if arguments["problem"]["which"] == "checkerboard":
        print_table_distribution(result)
    else:
        print_row_distribution(result)

    return result





def no_test(problem: TestableCombinatorialProblem,
            miner: FeatureMiner,
            runs: int,
            features_per_run: int):
    print(f"The generated problem is {problem}, more specifically \n{problem.long_repr()}")

    ideals = problem.get_ideal_features()
    for ideal in ideals:
        print(f"Ideal {problem.feature_repr(ideal.to_legacy_feature())}")
        print(miner.feature_selector.criterion.describe_feature(ideal, miner.feature_selector.ppi))

    def single_run():
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, features_per_run)
        print(f"The process took {execution_time} seconds")
        amount_of_found_ideals = len([mined for mined in mined_features if mined in ideals])
        print(f"(Found {amount_of_found_ideals} ideals) The features are:")
        for feature in mined_features:
            print(problem.feature_repr(feature.to_legacy_feature()))
            print(miner.feature_selector.criterion.describe_feature(feature, miner.feature_selector.ppi),
                  "(Present in ideals)" if feature in ideals else "")

    for _ in range(runs):
        single_run()

    return {"test_results": "you get to have lunch early!"}


def test_run_with_limited_budget(problem_parameters: dict,
                                 criterion_parameters: dict,
                                 miner_settings_list: list[dict],
                                 test_parameters: dict,
                                 budget: int) -> TestResults:
    problem: TestableCombinatorialProblem = TestingUtilities.decode_problem(problem_parameters)
    criterion = TestingUtilities.decode_criterion(criterion_parameters, problem)
    termination_predicate = run_with_limited_budget(budget)
    sample_size = test_parameters["sample_size"]
    features_per_run = test_parameters["features_per_run"]
    selector = TestingUtilities.make_selector(problem, sample_size, criterion)
    miners = [TestingUtilities.decode_miner(miner_args, selector, termination_predicate)
              for miner_args in miner_settings_list]

    def test_a_single_miner(miner: FeatureMiner) -> TestResults:
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, features_per_run)
        ideals = problem.get_ideal_features()
        amount_of_found_ideals = len([mined for mined in mined_features if mined in ideals])
        return {"miner": miner, "found": amount_of_found_ideals, "total": len(ideals), "time": execution_time}

    return {"results_for_each_miner": [test_a_single_miner(miner) for miner in miners]}


def test_budget_needed_to_find_ideals(problem_parameters: dict,
                                      criterion_parameters: dict,
                                      miner_settings_list: list[dict],
                                      test_parameters: dict,
                                      max_budget: int) -> TestResults:
    problem: TestableCombinatorialProblem = TestingUtilities.decode_problem(problem_parameters)
    criterion = TestingUtilities.decode_criterion(criterion_parameters, problem)
    termination_predicate = run_until_found_features(problem.get_ideal_features(), max_budget=max_budget)
    sample_size = test_parameters["sample_size"]
    features_per_run = test_parameters["features_per_run"]
    selector = TestingUtilities.make_selector(problem, sample_size, criterion)
    miners = [TestingUtilities.decode_miner(miner_args, selector, termination_predicate)
              for miner_args in miner_settings_list]

    def test_a_single_miner(miner: FeatureMiner) -> TestResults:
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, features_per_run)
        ideals = problem.get_ideal_features()
        amount_of_found_ideals = len([mined for mined in mined_features if mined in ideals])
        successfull = amount_of_found_ideals == len(ideals)
        used_budget = miner.feature_selector.used_budget
        return {"miner": miner, "successfull": successfull, used_budget: "used_budget", "time": execution_time}

    return {"results_for_each_miner": [test_a_single_miner(miner) for miner in miners]}


def test_compare_connectedness_of_results(problem_parameters: dict,
                                          criterion_parameters: dict,
                                          miner_settings_list: list[dict],
                                          test_parameters: dict,
                                          budget: int) -> TestResults:
    problem: GraphColouringProblem = TestingUtilities.decode_problem(problem_parameters)
    criterion = TestingUtilities.decode_criterion(criterion_parameters, problem)
    termination_predicate = run_with_limited_budget(budget)
    sample_size = test_parameters["sample_size"]
    features_per_run = test_parameters["features_per_run"]
    selector = TestingUtilities.make_selector(problem, sample_size, criterion)
    miners = [TestingUtilities.decode_miner(miner_args, selector, termination_predicate)
              for miner_args in miner_settings_list]


    def test_a_single_miner(miner: FeatureMiner) -> TestResults:
        def are_connected(node_index_a: int, node_index_b) -> bool:
            return bool(problem.adjacency_matrix[node_index_a, node_index_b])

        def count_edges(node_list: list[int]) -> int:
            return len([(node_a, node_b) for node_a, node_b in itertools.combinations(node_list, 2)
                        if are_connected(node_a, node_b)])

        def register_feature(feature_to_register: Feature, accumulator: defaultdict[int, list]):
            present_nodes = [var for var, val in feature_to_register.to_var_val_pairs()]
            amount_of_nodes = len(present_nodes)
            accumulator[amount_of_nodes].append(count_edges(present_nodes))

        edge_counts = defaultdict(list)

        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, features_per_run)
        for feature in mined_features:
            register_feature(feature, edge_counts)

        return {"miner": miner,
                "edge_counts": edge_counts,
                "runtime": execution_time}

    def sample_from_binomial_distribution(n: int, chance_of_success: float, samples: int) -> list[int]:
        def single_sample():
            return sum([random.random() < chance_of_success for _ in range(n)])

        return [single_sample() for _ in range(samples)]
    def test_simulated_miner() -> TestResults:
        samples_for_each_amount_of_nodes = 1000
        subgraph_sizes_to_simulate = range(problem.amount_of_nodes+1)
        def simulated_connections_for_subgraph_size(subgraph_size) -> list[int]:
            return sample_from_binomial_distribution(subgraph_size,
                                                     problem.chance_of_connection,
                                                     samples_for_each_amount_of_nodes)

        edge_counts = {subgraph_size: simulated_connections_for_subgraph_size(subgraph_size)
                       for subgraph_size in subgraph_sizes_to_simulate}

        return {"miner": "simulated",
                "edge_counts": edge_counts,
                "runtime": 0}  # just in case

    results = [test_a_single_miner(miner) for miner in miners]
    results.append(test_simulated_miner())
    return {"results_for_each_miner": results}



def apply_test_once(arguments: dict) -> dict:
    test_parameters = arguments["test"]
    test_kind = test_parameters["which"]

    if test_kind == "results_given_budget":
        return test_run_with_limited_budget(problem_parameters=arguments["problem"],
                                            criterion_parameters=arguments["criterion"],
                                            miner_settings_list=test_parameters["miners"],
                                            test_parameters=test_parameters,
                                            budget=test_parameters["budget"])
    elif test_kind == "budget_needed_to_find_ideals":
        return test_budget_needed_to_find_ideals(problem_parameters=arguments["problem"],
                                                 criterion_parameters=arguments["criterion"],
                                                 miner_settings_list=test_parameters["miners"],
                                                 test_parameters=test_parameters,
                                                 max_budget=test_parameters["budget"])
    elif test_kind == "budget_needed_to_find_ideals":
        return test_budget_needed_to_find_ideals(problem_parameters=arguments["problem"],
                                                 criterion_parameters=arguments["criterion"],
                                                 miner_settings_list=test_parameters["miners"],
                                                 test_parameters=test_parameters,
                                                 max_budget=test_parameters["budget"])

    if test_kind == "get_distribution":
        return test_get_distribution(arguments=arguments,
                                     runs=test_parameters["runs"],
                                     features_per_run=test_parameters["features_per_run"])
    elif test_kind == "no_test":
        problem, miner = generate_problem_miner(arguments)
        return no_test(problem=problem,
                       miner=miner,
                       runs=test_parameters["runs"],
                       features_per_run=test_parameters["features_per_run"])
    else:
        raise Exception("Test was not recognised")
