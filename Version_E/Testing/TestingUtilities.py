import json
import time
from json import JSONDecodeError

import numpy as np

from BenchmarkProblems.ArtificialProblem import ArtificialProblem
from BenchmarkProblems.BinVal import BinValProblem
from BenchmarkProblems.CheckerBoard import CheckerBoardProblem
from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem, TestableCombinatorialProblem
from BenchmarkProblems.GraphColouring import GraphColouringProblem
from BenchmarkProblems.Knapsack import KnapsackProblem
from BenchmarkProblems.OneMax import OneMaxProblem
from BenchmarkProblems.TrapK import TrapK
from SearchSpace import Candidate
from Version_E.BaselineAlgorithms import RandomSearch, HillClimber
from Version_E.BaselineAlgorithms.GA import GAMiner
from Version_E.BaselineAlgorithms.HillClimber import HillClimber
from Version_E.BaselineAlgorithms.RandomSearch import RandomSearch
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.ConstructiveMiner import ConstructiveMiner
from Version_E.InterestingAlgorithms.DestructiveMiner import DestructiveMiner
from Version_E.InterestingAlgorithms.Miner import FeatureMiner
from Version_E.InterestingAlgorithms.Miner import FeatureSelector
from Version_E.MeasurableCriterion.CriterionUtilities import All, Any, Not, Balance
from Version_E.MeasurableCriterion.Explainability import Explainability
from Version_E.MeasurableCriterion.GoodFitness import HighFitness, ConsistentFitness
from Version_E.MeasurableCriterion.Popularity import Overrepresentation, Commonality
from Version_E.MeasurableCriterion.Robustness import Robustness
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation

JSON = dict

from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion


def to_json_object(input) -> JSON:
    return json.loads(input)


def decode_problem(properties: JSON) -> CombinatorialProblem:
    problem_string = properties["which"]

    if problem_string == "binval":
        return BinValProblem(amount_of_bits=properties["size"],
                             base=properties["base"])
    elif problem_string == "onemax":
        return OneMaxProblem(amount_of_bits=properties["size"])
    elif problem_string == "trapk":
        return TrapK(amount_of_groups=properties["amount_of_groups"],
                     k=properties["k"])
    elif problem_string == "checkerboard":
        return CheckerBoardProblem(rows=properties["rows"],
                                   cols=properties["cols"])
    elif problem_string == "artificial":
        return ArtificialProblem(amount_of_bits=properties["size"],
                                 size_of_partials=properties["size_of_partials"],
                                 amount_of_features=properties["amount_of_features"],
                                 allow_overlaps=properties["allow_overlaps"])
    elif problem_string == "knapsack":
        return KnapsackProblem(expected_price=properties["expected_price"],
                               expected_weight=properties["expected_weight"],
                               expected_volume=properties["expected_volume"])
    elif problem_string == "graph":
        return GraphColouringProblem(amount_of_colours=properties["amount_of_colours"],
                                     amount_of_nodes=properties["amount_of_nodes"],
                                     chance_of_connection=properties["chance_of_connection"])
    else:
        raise Exception("The problem was not recognised")


def decode_criterion(properties: JSON, problem: CombinatorialProblem) -> MeasurableCriterion:
    criterion_string = properties["which"]

    if criterion_string == "explainability":
        return Explainability(problem)
    elif criterion_string == "high_fitness":
        return HighFitness()
    elif criterion_string == "low_fitness":
        return Not(HighFitness())
    elif criterion_string == "consistent_fitness":
        return ConsistentFitness()
    elif criterion_string == "overrepresentation":
        return Overrepresentation()
    elif criterion_string == "underrepresentation":
        return Not(Overrepresentation())
    elif criterion_string == "commonality":
        return Commonality()
    elif criterion_string == "robustness":
        return Robustness(properties["min_diff"], "max_diff")
    elif criterion_string == "not":
        return Not(decode_criterion(properties["argument"], problem))
    elif criterion_string == "all":
        parsed_arguments = [decode_criterion(criterion, problem) for criterion in properties["arguments"]]
        return All(parsed_arguments)
    elif criterion_string == "any":
        parsed_arguments = [decode_criterion(criterion, problem) for criterion in properties["arguments"]]
        return Any(parsed_arguments)
    elif criterion_string == "balance":
        parsed_arguments = [decode_criterion(criterion, problem) for criterion in properties["arguments"]]
        weights = properties["weights"]
        return Balance(parsed_arguments, weights=weights)


def decode_miner(properties: JSON, selector: FeatureSelector) -> FeatureMiner:
    """ converts a json string into an instance of a FeatureMiner object"""

    kind = properties["which"]

    if kind in "Constructive":
        return ConstructiveMiner(selector,
                                 stochastic=properties["stochastic"],
                                 at_most_parameters=properties["at_most"],
                                 amount_to_keep_in_each_layer=properties["amount_to_keep_in_each_layer"])
    elif kind == "Destructive":
        return DestructiveMiner(selector,
                                stochastic=properties["stochastic"],
                                at_least_parameters=properties["at_least"],
                                amount_to_keep_in_each_layer=properties["amount_to_keep_in_each_layer"])
    elif kind == "GA":
        return GAMiner(selector,
                       population_size=properties["population_size"],
                       iterations=properties["iterations"])
    elif kind == "Hill":
        return HillClimber(selector,
                           amount_to_generate=properties["population_size"])
    elif kind == "Random":
        return RandomSearch(selector,
                            amount_to_generate=properties["population_size"])


def get_random_candidates_and_fitnesses(problem: CombinatorialProblem,
                                        sample_size) -> (list[Candidate], list[float]):
    random_candidates = [problem.get_random_candidate_solution() for _ in range(sample_size)]

    scores = [problem.score_of_candidate(c) for c in random_candidates]
    return random_candidates, scores


def get_training_data(problem: CombinatorialProblem,
                      sample_size) -> PrecomputedPopulationInformation:
    training_samples = [problem.get_random_candidate_solution() for _ in range(sample_size)]
    fitness_list = [problem.score_of_candidate(c) for c in training_samples]
    return PrecomputedPopulationInformation(problem.search_space, training_samples, fitness_list)


def error_result(description) -> JSON:
    return {"error": description}


Seconds = float


def timed_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    return wrapper


def execute_and_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


def count_ideals_test(problem: TestableCombinatorialProblem, miner: FeatureMiner):
    ideals = problem.get_ideal_features()
    amount_to_consider = len(ideals) * 2
    found_features, execution_time = execute_and_time(miner.get_meaningful_features, amount_to_consider)
    found_features = set(found_features)
    return {"execution_time": execution_time,
            "presence": {f"{ideal}": (ideal in found_features) for ideal in ideals}}


def check_distribution_test(problem: CombinatorialProblem, miner):
    def register_features(feature: Feature, accumulator):
        mask_array = np.zeros(feature.variable_mask.tolist())
        accumulator += mask_array

    cell_coverings = np.zeros(problem.search_space.dimensions, dtype=int)
    found_features, execution_time = execute_and_time(miner.get_meaningful_features, 100)

    for feature in found_features:
        register_features(feature, cell_coverings)

    return {"execution_time": execution_time,
            "cell_position_counts": [count for count in cell_coverings]}


def apply_test(test_type: str, problem: CombinatorialProblem, miner: FeatureMiner) -> JSON:
    if test_type == "count_ideals":
        test_type_function = count_ideals_test
    elif test_type == "check_distribution":
        test_type_function = check_distribution_test
    else:
        return error_result("invalid test")

    data, execution_time = test_type_function(problem, miner)

    data["runtime"] = execution_time
    return data


def run_test(arguments_string) -> JSON:
    try:
        arguments_json = to_json_object(arguments_string)
    except JSONDecodeError:
        return error_result("JSON could not be parsed in problem settings")

    print(f"Received the arguments {arguments_json}")
    try:
        problem_object = decode_problem(arguments_json["problem"])
        criterion_object = decode_criterion(arguments_json["criterion"], problem_object)
    except KeyError:
        return error_result("missing parameter from JSON in problem settings")

    try:
        test_type = arguments_json["test"]
        sample_size = arguments_json["sample_size"]
        training_ppi = get_training_data(problem_object, sample_size)
        selector = FeatureSelector(training_ppi, criterion_object)
        miner_object = decode_miner(arguments_json["miner"], selector)
    except KeyError:
        return error_result("missing parameter from JSON in test settings")
    except JSONDecodeError:
        return error_result("JSON could not be parsed in test settings")

    result_json = apply_test(test_type, problem_object, miner_object)

    output_name = f"{test_type}~{sample_size}~{problem_object}~{miner_object}"

    output_json = {"parameters": arguments_json, "result": result_json}

    with open(output_name, 'w') as json_file:
        json.dump(output_json, json_file)
