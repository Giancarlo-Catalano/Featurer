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


def generate_problem_miner(arguments: dict) -> (CombinatorialProblem, FeatureMiner):
    problem = Problems.decode_problem(arguments["problem"])
    criterion = Criteria.decode_criterion(arguments["criterion"], problem)
    sample_size = arguments["sample_size"]
    training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)
    selector = FeatureSelector(training_ppi, criterion)
    miner = Miners.decode_miner(arguments["miner"], selector)
    return problem, miner


def check_successfullness(arguments: dict, runs: int, features_per_run: int) -> dict:
    def single_run() -> (int, int):
        print("Start of a single run")
        problem, miner = generate_problem_miner(arguments)
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, features_per_run)
        ideals = [Feature.from_legacy_feature(ideal, problem.search_space) for ideal in problem.get_ideal_features()]
        amount_of_found_ideals = len([mined for mined in mined_features if mined in ideals])
        return {"found": amount_of_found_ideals, "total": len(ideals)}

    def print_count_pairs(counts_json: dict):
        pairs = [(item["found"], item["total"]) for item in counts_json["test_results"]]
        for found, total in pairs:
            print(f"{found}\t{total}")

    result = {"test_results": [single_run() for _ in range(runs)]}
    print_count_pairs(result)
    return result


def check_distribution_test(problem: CombinatorialProblem, miner: FeatureMiner, runs: int,
                            features_per_run: int) -> dict:
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


def check_connectedness(arguments: dict, runs: int, features_per_run: int):
    def single_run():
        print("starting a single run")
        problem, miner = generate_problem_miner(arguments)

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

        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, features_per_run)
        for feature in mined_features:
            register_feature(feature, edge_counts)

        return {"edge_counts": edge_counts,
                "chance_of_connection": problem.chance_of_connection,
                "runtime": execution_time}

    return {"test_results": [single_run() for _ in range(runs)]}


def no_test(problem: BenchmarkProblems.CombinatorialProblem.TestableCombinatorialProblem,
            miner: FeatureMiner,
            runs: int,
            features_per_run: int):
    print(f"The generated problem is {problem}, more specifically \n{problem.long_repr()}")

    ideals = problem.get_ideal_features()
    ideals = [Feature.from_legacy_feature(ideal, problem.search_space)
              for ideal in ideals]
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


def apply_test(arguments: dict) -> dict:
    test_parameters = arguments["test"]
    test_kind = test_parameters["which"]

    if test_kind == "check_distribution":
        problem, miner = generate_problem_miner(arguments)
        return check_distribution_test(problem=problem,
                                       miner=miner,
                                       runs=test_parameters["runs"],
                                       features_per_run=test_parameters["features_per_run"])
    elif test_kind == "check_connectedness":
        return check_connectedness(arguments=arguments,
                                   runs=test_parameters["runs"],
                                   features_per_run=test_parameters["features_per_run"])
    elif test_kind == "check_successfullness":
        return check_successfullness(arguments=arguments,
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
