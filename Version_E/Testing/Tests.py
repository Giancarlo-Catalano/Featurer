import itertools
import json
import math
import random
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

import SearchSpace
import utils
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem, CombinatorialProblem
from BenchmarkProblems.GraphColouring import GraphColouringProblem
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, run_with_limited_budget, \
    run_until_found_features, FeatureSelector
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Sampling.GASampler import GASampler
from Version_E.Sampling.RegurgitationSampler import get_reference_features_for_regurgitation_sampling, \
    regurgitation_sample
from Version_E.Testing import TestingUtilities, Criteria
from Version_E.Testing.TestingUtilities import execute_and_time
from Version_E.Sampling.SimpleSampler import SimpleSampler, get_reference_features_for_simple_sampling

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


def test_get_distribution(arguments: Settings, runs: int,
                          features_per_run: int) -> TestResults:
    def register_feature(feature: Feature, accumulator):
        mask_array = np.array(feature.variable_mask.tolist())
        accumulator += mask_array

    def single_run():
        problem, miner = TestingUtilities.generate_problem_miner(arguments)
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


def get_miners_from_parameters(termination_predicate: TerminationPredicate,
                               problem: CombinatorialProblem,
                               criterion_parameters: dict,
                               miner_settings_list: list[dict],
                               test_parameters: dict) -> list[FeatureMiner]:
    criterion = TestingUtilities.decode_criterion(criterion_parameters, problem)
    sample_size = test_parameters["sample_size"]
    selector = TestingUtilities.make_selector(problem, sample_size, criterion)
    miners = [TestingUtilities.decode_miner(miner_args, selector, termination_predicate)
              for miner_args in miner_settings_list]
    return miners


def test_run_with_limited_budget(problem_parameters: dict,
                                 criterion_parameters: dict,
                                 miner_settings_list: list[dict],
                                 test_parameters: dict,
                                 budget: int) -> TestResults:
    problem: TestableCombinatorialProblem = TestingUtilities.decode_problem(problem_parameters)
    termination_predicate = run_with_limited_budget(budget)
    features_per_run = test_parameters["features_per_run"]

    miners = get_miners_from_parameters(termination_predicate, problem,
                                        criterion_parameters, miner_settings_list, test_parameters)

    def test_a_single_miner(miner: FeatureMiner, miner_parameters: Settings) -> TestResults:
        print(f"Testing {miner}")
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, features_per_run)
        mined_features = utils.remove_duplicates(mined_features, hashable=True)
        ideals = problem.get_ideal_features()
        amount_of_found_ideals = len([mined for mined in mined_features if mined in ideals])
        miner.feature_selector.reset_budget()
        return {"miner": miner_parameters,
                "found": amount_of_found_ideals,
                "total": len(ideals),
                "time": execution_time}

    return {"results_for_each_miner": [test_a_single_miner(miner, miner_parameters)
                                       for miner, miner_parameters in zip(miners, miner_settings_list)]}


def test_budget_needed_to_find_ideals(problem_parameters: dict,
                                      criterion_parameters: dict,
                                      miner_settings_list: list[dict],
                                      test_parameters: dict,
                                      max_budget: int) -> TestResults:
    problem: TestableCombinatorialProblem = TestingUtilities.decode_problem(problem_parameters)
    termination_predicate = run_until_found_features(problem.get_ideal_features(), max_budget=max_budget)
    features_per_run = test_parameters["features_per_run"]

    miners = get_miners_from_parameters(termination_predicate, problem,
                                        criterion_parameters, miner_settings_list, test_parameters)

    def test_a_single_miner(miner: FeatureMiner, miner_parameters: Settings) -> TestResults:
        print(f"Testing {miner}")
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, features_per_run)
        mined_features = utils.remove_duplicates(mined_features, hashable=True)
        ideals = problem.get_ideal_features()
        amount_of_found_ideals = len([mined for mined in mined_features if mined in ideals])
        successfull = amount_of_found_ideals == len(ideals)
        used_budget = miner.feature_selector.used_budget
        miner.feature_selector.reset_budget()
        return {"miner": miner_parameters,
                "successfull": successfull,
                "found": amount_of_found_ideals,
                "total": len(ideals),
                "used_budget": used_budget,
                "time": execution_time}

    return {"results_for_each_miner": [test_a_single_miner(miner, miner_parameters)
                                       for miner, miner_parameters in zip(miners, miner_settings_list)]}


def test_compare_connectedness_of_results(problem_parameters: dict,
                                          criterion_parameters: dict,
                                          miner_settings_list: list[dict],
                                          test_parameters: dict,
                                          max_budget: int) -> TestResults:
    problem: GraphColouringProblem = TestingUtilities.decode_problem(problem_parameters)
    termination_predicate = run_with_limited_budget(max_budget)
    features_per_run = test_parameters["features_per_run"]

    miners = get_miners_from_parameters(termination_predicate, problem,
                                        criterion_parameters, miner_settings_list, test_parameters)

    def test_a_single_miner(miner: FeatureMiner, miner_parameters: Settings) -> TestResults:
        print(f"Testing {miner}")

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

        miner.feature_selector.reset_budget()

        return {"miner": miner_parameters,
                "edge_counts": edge_counts,
                "time": execution_time}

    def sample_from_binomial_distribution(n: int, chance_of_success: float, samples: int) -> list[int]:
        def single_sample():
            return sum([random.random() < chance_of_success for _ in range(n)])

        return [single_sample() for _ in range(samples)]

    def test_simulated_miner() -> TestResults:
        samples_for_each_amount_of_nodes = 1000
        subgraph_sizes_to_simulate = range(problem.amount_of_nodes + 1)

        def simulated_connections_for_subgraph_size(subgraph_size) -> list[int]:
            return sample_from_binomial_distribution(subgraph_size,
                                                     problem.chance_of_connection,
                                                     samples_for_each_amount_of_nodes)

        edge_counts = {subgraph_size: simulated_connections_for_subgraph_size(subgraph_size)
                       for subgraph_size in subgraph_sizes_to_simulate}

        return {"miner": "simulated",
                "edge_counts": edge_counts,
                "runtime": 0}  # just in case

    results = [test_a_single_miner(miner, miner_parameters)
               for miner, miner_parameters in zip(miners, miner_settings_list)]
    # results.append(test_simulated_miner())
    return {"results_for_each_miner": results}



def test_compare_samplers(problem_parameters: dict,
                          fitness_criterion_parameters: dict,
                          reference_miner_parameters: dict,
                          sampling_miner_settings: dict,
                          ga_sampler_settings: dict,
                          test_parameters: dict,
                          max_budget: int) -> TestResults:
    problem: CombinatorialProblem = TestingUtilities.decode_problem(problem_parameters)
    termination_predicate = run_with_limited_budget(max_budget)
    amount_of_reference_features = test_parameters["amount_of_reference_features"]
    amount_of_sampled_candidates = test_parameters["amount_of_sampled_candidates"]
    fitness_criterion = Criteria.decode_criterion(fitness_criterion_parameters, problem)
    importance_of_explainability = test_parameters["importance_of_explainability"]

    termination_predicate = run_with_limited_budget(10000)
    sample_size = 2400
    training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)

    def sample_using_simple_sampler() -> list[SearchSpace.Candidate]:
        reference_features_for_simple_sampler = get_reference_features_for_simple_sampling(fitness_criterion=fitness_criterion,
                                                                                           problem = problem,
                                                                                           termination_predicate=termination_predicate,
                                                                                           ppi = training_ppi,
                                                                                           reference_miner_parameters = reference_miner_parameters,
                                                                                           amount_to_return=amount_of_reference_features,
                                                                                           importance_of_explainability=importance_of_explainability)
        temp_selector = FeatureSelector(training_ppi, fitness_criterion)
        scores_of_reference_features_for_simple_sampler = temp_selector.get_scores(
            reference_features_for_simple_sampler)
        simple_sampler = SimpleSampler(problem.search_space,
                                       list(zip(reference_features_for_simple_sampler,
                                                scores_of_reference_features_for_simple_sampler)))

        return [simple_sampler.sample_candidate() for _ in
                                                  range(amount_of_sampled_candidates)]


    def sample_using_regurgitation() -> list[SearchSpace.Candidate]:

        reference_features_for_regurgitation_sampler = get_reference_features_for_regurgitation_sampling(
            fitness_criterion=fitness_criterion,
            problem=problem,
            termination_predicate=termination_predicate,
            ppi=training_ppi,
            reference_miner_parameters=reference_miner_parameters,
            amount_to_return=amount_of_reference_features,
            importance_of_explainability=importance_of_explainability)



        return regurgitation_sample(reference_features=reference_features_for_regurgitation_sampler,
                                                                    fitness_criterion=fitness_criterion,
                                                                    termination_predicate=termination_predicate,  # might change in the future, could be faster!
                                                                    original_ppi=training_ppi,
                                                                    sampling_miner_parameters=sampling_miner_settings,
                                                                    amount_to_return=amount_of_sampled_candidates)


    def sample_using_ga() -> list[SearchSpace.Candidate]:

        ga_sampler = GASampler(fitness_function=problem.score_of_candidate,
                               search_space= problem.search_space,
                               population_size= ga_sampler_settings["population_size"],
                               generations= ga_sampler_settings["generations"])

        return ga_sampler.get_evolved_individuals(amount_of_sampled_candidates)

    sampled_from_simple = sample_using_simple_sampler()
    sampled_from_regurgitation = sample_using_regurgitation()
    sampled_using_ga = sample_using_ga()

    def fitnesses_from(samples: Iterable[SearchSpace.Candidate]) -> list[float]:
        return [problem.score_of_candidate(candidate) for candidate in samples]

    return {"from_simple": fitnesses_from(sampled_from_simple),
            "from_regurgitation": fitnesses_from(sampled_from_regurgitation),  # perhaps in the future there will be a normal GA?
            "from_ga": fitnesses_from(sampled_using_ga)
            }






def apply_test_once(arguments: Settings) -> TestResults:
    test_parameters = arguments["test"]
    test_kind = test_parameters["which"]

    if test_kind == "results_given_budget":
        miners = TestingUtilities.load_miners_from_second_command_line_argument()
        return test_run_with_limited_budget(problem_parameters=arguments["problem"],
                                            criterion_parameters=arguments["criterion"],
                                            miner_settings_list=miners,
                                            test_parameters=test_parameters,
                                            budget=test_parameters["budget"])
    elif test_kind == "budget_needed_to_find_ideals":
        miners = TestingUtilities.load_miners_from_second_command_line_argument()
        return test_budget_needed_to_find_ideals(problem_parameters=arguments["problem"],
                                                 criterion_parameters=arguments["criterion"],
                                                 miner_settings_list=miners,
                                                 test_parameters=test_parameters,
                                                 max_budget=test_parameters["budget"])
    elif test_kind == "compare_connectedness_of_results":
        miners = TestingUtilities.load_miners_from_second_command_line_argument()
        return test_compare_connectedness_of_results(problem_parameters=arguments["problem"],
                                                     criterion_parameters=arguments["criterion"],
                                                     miner_settings_list=miners,
                                                     test_parameters=test_parameters,
                                                     max_budget=test_parameters["budget"])

    if test_kind == "get_distribution":
        return test_get_distribution(arguments=arguments,
                                     runs=test_parameters["runs"],
                                     features_per_run=test_parameters["features_per_run"])
    elif test_kind == "no_test":
        problem, miner = TestingUtilities.generate_problem_miner(arguments)
        return no_test(problem=problem,
                       miner=miner,
                       runs=test_parameters["runs"],
                       features_per_run=test_parameters["features_per_run"])
    else:
        raise Exception("Test was not recognised")


def apply_test(arguments: Settings) -> list[TestResults]:
    runs: int = arguments["test"]["runs"]
    return [apply_test_once(arguments) for _ in range(runs)]


def run_test(arguments: dict):
    result_json = apply_test(arguments)  # important part
    output_json = {"parameters": arguments, "result": result_json}
    print(json.dumps(output_json))
