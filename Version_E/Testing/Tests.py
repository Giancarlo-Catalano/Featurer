import itertools
import json
import math
import random
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

import utils
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem, CombinatorialProblem
from BenchmarkProblems.GraphColouring import GraphColouringProblem, InsularGraphColouringProblem
from BenchmarkProblems.TrapK import TrapK
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, run_with_limited_budget, \
    run_until_found_features, FeatureSelector, run_until_fixed_amount_of_iterations
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Sampling.EDAs.EDASampler import UMDA
from Version_E.Sampling.FullSolutionSampler import FullSolutionSampler
from Version_E.Sampling.GASampler import GASampler
from Version_E.Sampling.SimpleSampler import SimpleSampler
from Version_E.Testing import TestingUtilities
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
    ga_budget = test_parameters["ga_budget"]
    selector = TestingUtilities.make_selector(problem, sample_size, ga_budget, criterion)
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

    def count_results(found_features: Iterable[Feature]) -> (int, int):
        if isinstance(problem, InsularGraphColouringProblem):
            amount_of_found = problem.count_how_many_islets_covered(found_features)
            amount_of_total = problem.amount_of_islets
            return amount_of_found, amount_of_total
        elif isinstance(problem, TrapK):
            ideal_presence_matrix = np.array([problem.contains_which_ideal_features(feature)
                                              for feature in found_features])
            ideal_presence_array = np.any(ideal_presence_matrix, axis=0)
            amount_of_covered_groups = int(np.sum(ideal_presence_array))
            return amount_of_covered_groups, problem.amount_of_groups
        else:
            ideals = problem.get_ideal_features()
            amount_of_found_ideals = len([ideal for ideal in ideals if ideal in found_features])
            return amount_of_found_ideals, len(ideals)

    def test_a_single_miner(miner: FeatureMiner, miner_parameters: Settings) -> TestResults:
        # print(f"Testing {miner}")
        mined_features, execution_time = execute_and_time(miner.get_meaningful_features, features_per_run)
        amount_of_found_features, total_possible = count_results(mined_features)
        miner.feature_selector.reset_budget()
        return {"miner": miner_parameters,
                "found": amount_of_found_features,
                "total": total_possible,
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
        # print(f"Testing {miner}")
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


def test_performance_given_bootstrap(problem_parameters: dict,
                                     criterion_parameters: dict,
                                     miner_parameters: dict,
                                     test_parameters: dict) -> TestResults:
    problem: TestableCombinatorialProblem = TestingUtilities.decode_problem(problem_parameters)
    problem_str = problem_parameters["problem"]["which"]  # assumes it's been shuffled
    mined_per_run = test_parameters["features_per_run"]
    miner_max_budget = test_parameters["miner_max_budget"]
    ideal_features = problem.get_ideal_features()
    found_all_targets = run_until_found_features(ideal_features, miner_max_budget)

    generations_for_bootstrap = test_parameters["bootstrap_generations"]
    populations_for_bootstrap = test_parameters["bootstrap_populations"]

    def test_for_population_size(pop_size: int):
        # first we make the ga sampler, but we have to give it a dummy termination criteria
        bogus_termination_criteria = run_with_limited_budget(1)
        ga_sampler = GASampler(fitness_function=problem.score_of_candidate,
                               search_space=problem.search_space,
                               population_size=pop_size,
                               termination_criteria=bogus_termination_criteria)

        # this will be the reference population, which will be reused over and over
        # Note that when this is None, ga_miner knows to make a random population
        bootstrap_population = None

        for generations in generations_for_bootstrap:
            # update the reference population by running the ga to the desired amount of generations
            ga_sampler.termination_criteria = run_until_fixed_amount_of_iterations(generations)
            bootstrap_population = ga_sampler.evolve_population(population=bootstrap_population)

            # make a ppi out of that population, and all the related objects
            bootstrap_ppi = PrecomputedPopulationInformation.from_preevolved_population(bootstrap_population, problem)
            criterion = TestingUtilities.decode_criterion(criterion_parameters, problem)
            selector = FeatureSelector(bootstrap_ppi, criterion)

            # use the reference population to mine features
            miner = TestingUtilities.decode_miner(miner_parameters,
                                                  selector,
                                                  termination_predicate=found_all_targets)
            mined_features, execution_time = execute_and_time(miner.get_meaningful_features, mined_per_run)
            mined_features = utils.remove_duplicates(mined_features, hashable=True)

            # check whether the run was successfull or not
            amount_of_found_ideals = len([mined for mined in mined_features if mined in ideal_features])
            successful = amount_of_found_ideals == len(ideal_features)

            # measure how much budget was used
            used_budget_by_miner = miner.feature_selector.used_budget

            yield {"bootstrap_population_size": pop_size,
                   "bootstrap_generations": generations,
                   "problem_str": problem_str,
                   "successful": successful,
                   "used_miner_budget": used_budget_by_miner,
                   "time": execution_time}

    all_results = [item_for_combination
                   for bootstrap_pop_size in populations_for_bootstrap
                   for item_for_combination in test_for_population_size(bootstrap_pop_size)]

    return {"results_for_params": all_results}


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
        # print(f"Testing {miner}")

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


def test_compare_samplers_old(problem_parameters_list: list[dict],
                              sampler_list: list[dict],
                              criterion_parameters: dict,
                              test_parameters: dict) -> TestResults:
    def get_results_for_problem(problem_parameters: dict) -> list[dict]:
        problem: TestableCombinatorialProblem = TestingUtilities.decode_problem(problem_parameters)
        problem_str: str = problem_parameters["problem"]["which"]
        global_optima_fitness = problem.get_global_optima_fitness()

        # generate the reference population using a ga
        reference_population_generator = GASampler(fitness_function=problem.score_of_candidate,
                                                   search_space=problem.search_space,
                                                   population_size=test_parameters["reference_population_size"],
                                                   termination_criteria=run_with_limited_budget(
                                                       test_parameters["reference_budget"]))

        reference_population = reference_population_generator.evolve_population()
        reference_ppi = PrecomputedPopulationInformation.from_preevolved_population(reference_population, problem)
        criterion = TestingUtilities.decode_criterion(criterion_parameters, problem)
        selector = FeatureSelector(reference_ppi, criterion)

        full_solution_amount = test_parameters["full_solution_amount"]

        # test the various ways of obtaining a full solution
        #    Miner gives PSs -> PSs are passed to a SimpleSampler -> Full solutions
        #    EDA or Traditional GA -> Full Solutions

        fitness_function = problem.score_of_candidate

        def get_miner_sampler(sampler_dict: dict) -> SimpleSampler:
            miner = TestingUtilities.decode_miner(sampler_dict["miner"],
                                                  selector,
                                                  run_with_limited_budget(sampler_dict["miner_budget"]))
            basis_features = miner.get_meaningful_features(sampler_dict["basis_PS_amount"])
            scored_basis_features = list(zip(basis_features, selector.get_scores(basis_features)))
            return SimpleSampler(search_space=problem.search_space,
                                 features_with_scores=scored_basis_features,
                                 fitness_function=fitness_function)

        def get_baseline_sampler(sampler_dict: dict) -> FullSolutionSampler:
            sampler_kind = sampler_dict["which"]
            if sampler_kind == "GA":
                return GASampler(fitness_function=fitness_function,
                                 search_space=problem.search_space,
                                 population_size=sampler_dict["population_size"],
                                 termination_criteria=run_with_limited_budget(sampler_dict["budget"]))
            elif sampler_kind == "UMDA":
                return UMDA(search_space=problem.search_space,
                            fitness_function=fitness_function,
                            population_size=sampler_dict["population_size"],
                            termination_criteria=run_with_limited_budget(sampler_dict["budget"]))
            else:
                raise Exception("Sampler method was not recognised")

        def get_datapoints_for_sampler(sampler_dict: dict) -> list[float]:
            method = sampler_dict["method"]
            sampler = get_miner_sampler(sampler_dict) if method == "via_miner" else get_baseline_sampler(sampler_dict)
            sampled_with_scores = sampler.sample(full_solution_amount)
            sampled, scores = utils.unzip(sampled_with_scores)
            scores = [float(score) for score in scores]  # apparently these values are not json serializable otherwise ?
            return scores

        def result_item_for_sampler(sampler_dict: dict) -> dict:
            fitnesses = get_datapoints_for_sampler(sampler_dict)
            contains_global_optima = global_optima_fitness in fitnesses
            return {"sampler": sampler_dict,
                    "problem": problem_str,
                    "fitnesses": fitnesses,
                    "successfull": contains_global_optima}

        return [result_item_for_sampler(sampler_dict) for sampler_dict in sampler_list]

    return {"sampling_results": [result_item for problem_dict in problem_parameters_list
                                 for result_item in get_results_for_problem(problem_dict)]}


def test_sampling_aux(problem_params: dict,
                      method_params: dict,
                      total_evaluation_budget: int,
                      arguments: dict) -> list[dict]:
    problem: TestableCombinatorialProblem = TestingUtilities.decode_problem(problem_params)
    problem_str: str = problem_params["problem"]["which"]
    global_optima_fitness = problem.get_global_optima_fitness()


    amount_of_full_solutions_to_generate = arguments["test"]["fs_return_amount"]

    def get_miner_samplers(method_params: dict) -> Iterable[SimpleSampler]:
        def get_ref_pop_ppi(normal_eval_budget: int) -> PrecomputedPopulationInformation:
            reference_population = [problem.get_random_candidate_solution() for _ in range(normal_eval_budget)]
            fitnesses = [problem.score_of_candidate(c) for c in reference_population]
            return PrecomputedPopulationInformation(problem.search_space, reference_population, fitnesses)

        criterion = TestingUtilities.decode_criterion(arguments["criterion"], problem)

        def get_miner(ps_budget: int) -> FeatureMiner:
            miner_params = method_params["miner"]
            fs_budget = total_evaluation_budget - ps_budget
            reference_ppi = get_ref_pop_ppi(fs_budget)
            selector = FeatureSelector(reference_ppi, criterion)
            termination_predicate = run_with_limited_budget(ps_budget)
            return TestingUtilities.decode_miner(miner_params, selector, termination_predicate)

        def get_sampler_from_miner(miner: FeatureMiner) -> SimpleSampler:
            basis_size = method_params["basis_size"]
            basis = miner.get_meaningful_features(basis_size)
            basis_scores = miner.feature_selector.get_scores(basis)
            return SimpleSampler(search_space=miner.search_space,
                                 features_with_scores=list(zip(basis, basis_scores)),
                                 fitness_function=problem.score_of_candidate)

        ps_budgets = np.ceil(np.array(method_params["ps_budgets_proportions"])*total_evaluation_budget).astype(int)
        return map(get_sampler_from_miner, map(get_miner, ps_budgets))

    def get_datapoints_from_sampler(sampler: FullSolutionSampler) -> list[float]:
        sampled_with_scores = sampler.sample(amount_of_full_solutions_to_generate)
        sampled, scores = utils.unzip(sampled_with_scores)
        scores = [float(score) for score in scores]  # apparently these values are not json serializable otherwise ?
        return scores

    sampler_kind = method_params["method"]

    if sampler_kind == "via_miner":
        samplers = get_miner_samplers(method_params)
        for sampler, ps_evals in zip(samplers, method_params["ps_budgets_proportions"]):
            scores = get_datapoints_from_sampler(sampler)
            contains_global_optima = global_optima_fitness in scores
            yield {"method": method_params["method"],
                   "normal_eval_budget": total_evaluation_budget,
                   "ps_eval_budget_prop": ps_evals,
                   "problem": problem_str,
                   "fitnesses": scores,
                   "contains_global_optima": contains_global_optima}
    else:
        if sampler_kind == "GA":
            sampler = GASampler(fitness_function=problem.score_of_candidate,
                                search_space=problem.search_space,
                                population_size=method_params["population_size"],
                                termination_criteria=run_with_limited_budget(total_evaluation_budget))
        elif sampler_kind == "UMDA":
            sampler = UMDA(search_space=problem.search_space,
                           fitness_function=problem.score_of_candidate,
                           population_size=method_params["population_size"],
                           termination_criteria=run_with_limited_budget(total_evaluation_budget))
        else:
            raise Exception("Sampler method was not recognised")

        scores = get_datapoints_from_sampler(sampler)
        contains_global_optima = global_optima_fitness in scores
        yield {"method": method_params["method"]+f"{method_params['population_size']}",
               "normal_eval_budget": total_evaluation_budget,
               "problem": problem_str,
               "fitnesses": scores,
               "contains_global_optima": contains_global_optima}


def test_compare_samplers(problems_params: list[dict],
                          methods_params: list[dict],
                          budget_list: list[int],
                          arguments: dict) -> TestResults:
    return {"sampling_results": [result_item
                                 for p in problems_params
                                 for m in methods_params
                                 for b in budget_list
                                 for result_item in test_sampling_aux(p, m, b, arguments)]}


def apply_test_once(arguments: Settings) -> TestResults:
    test_parameters = arguments["test"]
    test_kind = test_parameters["which"]

    if test_kind == "results_given_budget":
        miners = TestingUtilities.load_data_from_second_file()
        return test_run_with_limited_budget(problem_parameters=arguments["problem"],
                                            criterion_parameters=arguments["criterion"],
                                            miner_settings_list=miners,
                                            test_parameters=test_parameters,
                                            budget=test_parameters["budget"])
    elif test_kind == "budget_needed_to_find_ideals":
        miners = TestingUtilities.load_data_from_second_file()
        return test_budget_needed_to_find_ideals(problem_parameters=arguments["problem"],
                                                 criterion_parameters=arguments["criterion"],
                                                 miner_settings_list=miners,
                                                 test_parameters=test_parameters,
                                                 max_budget=test_parameters["max_budget"])
    elif test_kind == "compare_connectedness_of_results":
        miners = TestingUtilities.load_data_from_second_file()
        return test_compare_connectedness_of_results(problem_parameters=arguments["problem"],
                                                     criterion_parameters=arguments["criterion"],
                                                     miner_settings_list=miners,
                                                     test_parameters=test_parameters,
                                                     max_budget=test_parameters["budget"])
    elif test_kind == "compare_samplers":
        samplers: list[dict] = TestingUtilities.load_data_from_second_file()
        problems: list[dict] = arguments["problems"]
        return test_compare_samplers(problems_params=problems,
                                     methods_params=samplers,
                                     budget_list=arguments["test"]["fs_budgets"],
                                     arguments=arguments)

    if test_kind == "get_distribution":
        return test_get_distribution(arguments=arguments,
                                     runs=test_parameters["runs"],
                                     features_per_run=test_parameters["features_per_run"])
    if test_kind == "budget_given_bootstrap":
        miners = TestingUtilities.load_data_from_second_file()
        return test_performance_given_bootstrap(problem_parameters=arguments["problem"],
                                                miner_parameters=test_parameters["miner"],
                                                criterion_parameters=arguments["criterion"],
                                                test_parameters=test_parameters)
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
