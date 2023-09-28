import time

import numpy as np

from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem, CombinatorialProblem
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner


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

    print("The ideals are\n"+"\n".join(f"{ideal}" for ideal in ideals))

    def single_run():
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, amount_to_consider)
        mined_features = [Feature.from_legacy_feature(old_feature, problem.search_space)
                          for old_feature in mined_features]

        print("The Features are\n" + "\n".join(f"{feature}" for feature in mined_features))

        print("The found")
        amount_of_found_ideals = len([ideal for ideal in ideals if ideal in mined_features])
        return {"found_ideals": amount_of_found_ideals, "total_ideals": total_ideals, "runtime": execution_time}

    return {"test_results": [single_run() for _ in range(runs)]}


def check_distribution_test(problem: CombinatorialProblem, miner: FeatureMiner, runs: int):
    def register_feature(feature: Feature, accumulator):
        mask_array = np.zeros(feature.variable_mask.tolist())
        accumulator += mask_array

    def single_run():
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, 100)
        cell_coverings = np.zeros(problem.search_space.dimensions, dtype=int)
        for feature in mined_features:
            register_feature(feature, cell_coverings)
            return {"position_counts": [int(count) for count in cell_coverings],
                    "runtime": execution_time}

    return {"test_results": [single_run() for _ in range(runs)]}


def apply_test(test_parameters: dict, problem: CombinatorialProblem, miner: FeatureMiner) -> dict:
    test_type = test_parameters["which"]
    runs = test_parameters["runs"]
    if test_type == "count_ideals":
        return count_ideals_test(problem, miner, runs)
    elif test_type == "check_distribution":
        return check_distribution_test(problem, miner, runs)
    else:
        raise Exception("Test was not recognised")





