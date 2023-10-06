import itertools
import time

import numpy as np

import BenchmarkProblems.GraphColouring
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem, CombinatorialProblem
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, FeatureSelector
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Testing import Problems, Criteria, Miners


def execute_and_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


def run_multiple_times(func, runs, *args, **kwargs):
    return [func(*args, **kwargs) for _ in range(runs)]


def count_ideals_test(problem: TestableCombinatorialProblem, miner: FeatureMiner, runs: int) -> dict:
    ideals = problem.get_ideal_features()
    ideals = [Feature.from_legacy_feature(ideal, problem.search_space)
              for ideal in ideals]
    amount_to_consider = len(ideals) * 2
    total_ideals = len(ideals)

    print("The ideals are\n" + "\n".join(f"{ideal}" for ideal in ideals))

    def single_run():
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, amount_to_consider)

        print("The Features are\n" + "\n".join(f"{feature}" for feature in mined_features))

        print("The found")
        amount_of_found_ideals = len([ideal for ideal in ideals if ideal in mined_features])
        return {"found_ideals": amount_of_found_ideals, "total_ideals": total_ideals, "runtime": execution_time}

    return {"test_results": [single_run() for _ in range(runs)]}


def check_distribution_test(problem: CombinatorialProblem, miner: FeatureMiner, runs: int, features_per_run: int) -> dict:
    def register_feature(feature: Feature, accumulator):
        mask_array = np.array(feature.variable_mask.tolist())
        accumulator += mask_array

    def single_run():
        print("Start of a run")
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, features_per_run)
        cell_coverings = np.zeros(problem.search_space.dimensions, dtype=int)
        for feature in mined_features:
            register_feature(feature, cell_coverings)
        return {"position_counts": [int(count) for count in cell_coverings],
                "runtime": execution_time}

    return {"test_results": [single_run() for _ in range(runs)]}


def print_table_for_distribution(distribution_json: dict, original_problem):
    counts = np.array([item["position_counts"] for item in distribution_json["test_results"]])

    counts = np.sum(counts, axis=0)
    counts = counts.reshape((original_problem.rows, -1))

    for row in counts:
        print("\t".join([f"{item}" for item in row]))


def print_row_for_distribution(distribution_json: dict, original_problem):
    counts = np.array([item["position_counts"] for item in distribution_json["test_results"]])
    counts = np.sum(counts, axis=0)
    print("\t".join([f"{item}" for item in counts]))


def check_linkage(problem: BenchmarkProblems.GraphColouring.GraphColouringProblem, miner: FeatureMiner, runs: int):
    def register_feature(feature: Feature, accumulator):
        used_vars = np.array(feature.variable_mask.tolist(), dtype=float)
        accumulator += np.outer(used_vars, used_vars)

    def are_connected(node_index_a: int, node_index_b) -> bool:
        return bool(problem.adjacency_matrix[node_index_a, node_index_b])

    def single_run():
        cooccurrence_matrix = np.zeros((problem.amount_of_nodes, problem.amount_of_nodes), dtype=float)
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, 100)
        for feature in mined_features:
            register_feature(feature, cooccurrence_matrix)

        cooccurrence_matrix /= len(mined_features)

        connected_values = []
        unconnected_values = []
        pairs_to_inspect = [(row, col)
                            for row in range(problem.amount_of_nodes)
                            for col in range(problem.amount_of_nodes)
                            if col > row]

        for row, col in pairs_to_inspect:
            value = cooccurrence_matrix[row, col]
            (connected_values if are_connected(row, col) else unconnected_values).append(value)

        return {"connected_node_cooccurrence_proportion": connected_values,
                "unconnected_node_cooccurence_proportion": unconnected_values,
                "runtime": execution_time}

    return {"test_results": [single_run() for _ in range(runs)]}


def check_connectedness(arguments: dict, runs: int):

    def single_run():
        print("starting a single run")
        problem: BenchmarkProblems.GraphColouring.GraphColouringProblem = Problems.decode_problem(arguments["problem"])
        criterion = Criteria.decode_criterion(arguments["criterion"], problem)
        sample_size = arguments["sample_size"]
        training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)

        selector = FeatureSelector(training_ppi, criterion)
        miner = Miners.decode_miner(arguments["miner"], selector)

        def are_connected(node_index_a: int, node_index_b) -> bool:
            return bool(problem.adjacency_matrix[node_index_a, node_index_b])

        def count_edges(node_list: list[int]) -> int:
            return len([(node_a, node_b) for node_a, node_b in itertools.combinations(node_list, 2)
                        if are_connected(node_a, node_b)])

        def register_feature(feature: Feature, accumulator: list[(int, int)]):
            present_nodes = [var for var, val in feature.to_var_val_pairs()]
            amount_of_nodes = len(present_nodes)
            accumulator.append((amount_of_nodes, count_edges(present_nodes)))

        edge_counts = []

        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, 100)
        for feature in mined_features:
            register_feature(feature, edge_counts)

        return {"edge_counts": edge_counts,
                "chance_of_connection": problem.chance_of_connection,
                "runtime": execution_time}

    return {"test_results": [single_run() for _ in range(runs)]}


def no_test(problem: BenchmarkProblems.CombinatorialProblem.TestableCombinatorialProblem, miner: FeatureMiner, runs: int):
    print(f"The generated problem is {problem}, more specifically \n{problem.long_repr()}")

    ideals = problem.get_ideal_features()
    ideals = [Feature.from_legacy_feature(ideal, problem.search_space)
              for ideal in ideals]
    for ideal in ideals:
        print(f"Ideal {problem.feature_repr(ideal.to_legacy_feature())}")
        print(miner.feature_selector.criterion.describe_feature(ideal, miner.feature_selector.ppi))
    def single_run():
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, 100)
        print(f"The process took {execution_time} seconds")
        print("The features are:")
        for feature in mined_features:
            print(problem.feature_repr(feature.to_legacy_feature()))
            print(miner.feature_selector.criterion.describe_feature(feature, miner.feature_selector.ppi))

    for _ in range(runs):
        single_run()

    return {"test_results": "you get to have lunch early!"}


def apply_test(test_parameters: dict, problem: CombinatorialProblem, miner: FeatureMiner, arguments: dict) -> dict:
    test_type = test_parameters["which"]
    runs = test_parameters["runs"]
    if test_type == "count_ideals":
        return count_ideals_test(problem, miner, runs)
    elif test_type == "check_distribution":
        features_per_run = test_parameters["features_per_run"]
        print_row_for_distribution(check_distribution_test(problem, miner, runs, features_per_run), problem)
    elif test_type == "check_linkage":
        return check_linkage(problem, miner, runs)
    elif test_type == "check_connectedness":
        return check_connectedness(arguments, runs)
    elif test_type == "no_test":
        return no_test(problem, miner, runs)
    else:
        raise Exception("Test was not recognised")



