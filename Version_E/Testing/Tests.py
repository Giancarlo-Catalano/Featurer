import itertools
import json
import math
import time
import random

import numpy as np

import BenchmarkProblems.GraphColouring
import utils
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem, CombinatorialProblem
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, FeatureSelector
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Testing import Problems, Criteria, Miners
from collections import defaultdict

Settings = dict
TestResults = dict


def execute_and_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


def run_multiple_times(func, runs, *args, **kwargs):
    return [func(*args, **kwargs) for _ in range(runs)]


def generate_problem_miner(arguments: Settings, miner_arguments=None) -> (CombinatorialProblem, FeatureMiner):
    problem = Problems.decode_problem(arguments["problem"])
    criterion = Criteria.decode_criterion(arguments["criterion"], problem)
    sample_size = arguments["sample_size"]
    training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)
    selector = FeatureSelector(training_ppi, criterion)

    if miner_arguments is None:
        miner_arguments = arguments["miner"]
    miner = Miners.decode_miner(miner_arguments, selector)
    return problem, miner


def check_successfullness(arguments: Settings, runs: int, features_per_run: int, miner_arguments=None,
                          debug=False) -> TestResults:
    def single_run() -> (int, int):
        print("Start of a single run")
        problem, miner = generate_problem_miner(arguments, miner_arguments=miner_arguments)
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, features_per_run)
        ideals = [Feature.from_legacy_feature(ideal, problem.search_space) for ideal in problem.get_ideal_features()]
        amount_of_found_ideals = len([mined for mined in mined_features if mined in ideals])
        return {"found": amount_of_found_ideals, "total": len(ideals)}

    def print_count_pairs(counts_json: dict):
        pairs = [(item["found"], item["total"]) for item in counts_json["test_results"]]
        for found, total in pairs:
            print(f"{found}\t{total}")

    result = {"test_results": [single_run() for _ in range(runs)]}
    if debug:
        print_count_pairs(result)

    return result


def check_distribution_test(arguments: Settings, runs: int,
                            features_per_run: int) -> TestResults:
    def register_feature(feature: Feature, accumulator):
        mask_array = np.array(feature.variable_mask.tolist())
        accumulator += mask_array

    def single_run():
        print("Start of a run")
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


def make_csv_for_successes(input_name, output_name: str):
    with open(input_name, "r") as input_file:
        data = json.load(input_file)
        miners_and_results = [(item["miner"], item["result"]) for item in data["result"]["test_results"]]

        def convert_pairs(pairs_list):
            return [item["found"] for item in pairs_list]

        miners_and_results = [(miner, convert_pairs(result)) for miner, result in miners_and_results]

        with open(output_name, "w") as output_file:
            for miner, results in miners_and_results:
                output_file.write(f"{miner}\t" + '\t'.join(f"{value}" for value in results))


def make_csv_for_connectedness(input_name, output_name: str):
    def merge_dicts(dicts: list[dict]):
        def items_for_key(key):
            recovered_items = [single_dict.setdefault(key, []) for single_dict in dicts]
            return utils.concat_lists(recovered_items)

        all_keys = {key for single_dict in dicts
                    for key in single_dict.keys()}
        return {key: items_for_key(key) for key in all_keys}

    with open(input_name, "r") as input_file:
        data = json.load(input_file)
        trials = [item["edge_counts"] for item in data["result"]["test_results"]]
        baselines = [item["binomial_distributions"] for item in data["result"]["test_results"]]

        # aggregate different trial runs
        merged_trials = merge_dicts(trials)
        merged_baselines = merge_dicts(baselines)

        # sort by amount of nodes
        merged_trials = dict(sorted(merged_trials.items(), key=utils.first))
        merged_baselines = dict(sorted(merged_baselines.items(), key=utils.first))

        with open(output_name, "w") as output_file:
            for (observed_key, observed_values), (expected_key, expected_values) in zip(merged_trials.items(),
                                                                                        merged_baselines.items()):
                output_file.write(f"Observed_{observed_key}")
                output_file.write("".join([f"\t{item}" for item in observed_values]))
                output_file.write("\n")
                output_file.write(f"Expected_{expected_key}")
                output_file.write("".join([f"\t{item}" for item in expected_values]))
                output_file.write("\n")


def check_connectedness(arguments: Settings, runs: int, features_per_run: int,
                        with_binomial_distribution=False) -> TestResults:
    def single_run():
        print("starting a single run")
        problem, miner = generate_problem_miner(arguments)

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

        samples_for_each = runs ** 2  # arbitrary

        binomial_distributions = {node_amount: sample_from_binomial_distribution(int(node_amount * node_amount / 2),
                                                                                 problem.chance_of_connection,
                                                                                 samples_for_each)
                                  for node_amount in edge_counts.keys()}

        return {"edge_counts": edge_counts,
                "chance_of_connection": problem.chance_of_connection,
                "binomial_distributions": binomial_distributions,
                "runtime": execution_time}

    return {"test_results": [single_run() for _ in range(runs)]}


def sample_from_binomial_distribution(n: int, chance_of_success: float, samples: int) -> list[int]:
    def single_sample():
        return sum([random.random() < chance_of_success for _ in range(n)])

    return [single_sample() for _ in range(samples)]


def check_miners(arguments: Settings, features_per_run: int, runs_per_miner: int,
                 miners_settings_list: list[Settings]) -> TestResults:
    def single_run(miner_arguments) -> dict:
        print(f"executing {miner_arguments}")
        results = check_successfullness(arguments,
                                        runs=runs_per_miner,
                                        features_per_run=features_per_run,
                                        miner_arguments=miner_arguments)

        return {"miner": miner_arguments,
                "result": results["test_results"]}

    return {"test_results": [single_run(miner_arguments) for miner_arguments in miners_settings_list]}


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
        return check_distribution_test(arguments,
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
    elif test_kind == "check_miners":
        return check_miners(arguments,
                            runs_per_miner=test_parameters["runs"],
                            features_per_run=test_parameters["features_per_run"],
                            miners_settings_list=test_parameters["miners"])
    elif test_kind == "no_test":
        problem, miner = generate_problem_miner(arguments)
        return no_test(problem=problem,
                       miner=miner,
                       runs=test_parameters["runs"],
                       features_per_run=test_parameters["features_per_run"])
    else:
        raise Exception("Test was not recognised")
