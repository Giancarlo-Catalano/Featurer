import struct
from enum import Enum, auto
from typing import Callable

from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from BenchmarkProblems.BinVal import BinValProblem
from BenchmarkProblems.OneMax import OneMaxProblem
from BenchmarkProblems.ArtificialProblem import ArtificialProblem
from BenchmarkProblems.Knapsack import KnapsackProblem
from BenchmarkProblems.GraphColouring import GraphColouringProblem
from BenchmarkProblems.CheckerBoard import CheckerBoardProblem
from BenchmarkProblems.TrapK import TrapK
import utils
from Version_E.InterestingAlgorithms.ConstructiveMiner import ConstructiveMiner
from Version_E.InterestingAlgorithms.DestructiveMiner import DestructiveMiner
from Version_E.BaselineAlgorithms.GA import GAMiner
from Version_E.BaselineAlgorithms.RandomSearch import RandomSearch
from Version_E.BaselineAlgorithms.HillClimber import HillClimber
from Version_E.BaselineAlgorithms import GA, RandomSearch, HillClimber
from Version_E.InterestingAlgorithms.Miner import FeatureMiner
from Version_E.MeasurableCriterion.CriterionUtilities import All, Any, Not, Balance
from Version_E.MeasurableCriterion.GoodFitness import HighFitness, ConsistentFitness
from Version_E.MeasurableCriterion.Explainability import Explainability
from Version_E.MeasurableCriterion.Popularity import Overrepresentation, Commonality
from Version_E.InterestingAlgorithms.Miner import FeatureSelector
from Version_E.MeasurableCriterion.Robustness import Robustness

import json

from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion


def to_json_object(input) -> dict:
    return json.loads(input)


def decode_problem(properties: dict) -> CombinatorialProblem:
    problem_string = properties["problem"]

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
                                   cols = properties["cols"])
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

def decode_criterion(properties: dict, problem: CombinatorialProblem) -> MeasurableCriterion:
    criterion_string = properties["criterion"]

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
        return Robustness(properties["min_diff"],"max_diff")
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


def decode_algorithm(properties: dict, selector: FeatureSelector) -> FeatureMiner:
    """ converts a json string into an instance of a FeatureMiner object"""

    kind = properties["kind"]
    population_size = properties["popsize"]


    if kind in "Constructive":
        return ConstructiveMiner(selector,
                                 stochastic=properties["stochastic"],
                                 at_most_parameters = properties["at_most"],
                                 amount_to_keep_in_each_layer= properties["amount_to_keep_in_each_layer"])
    elif kind == "Destructive":
        return  DestructiveMiner(selector,
                                 stochastic=properties["stochastic"],
                                 at_least_parameters=properties["at_least"],
                                 amount_to_keep_in_each_layer=properties["amount_to_keep_in_each_layer"])
    elif kind == "GA":
        return GAMiner(selector,
                       population_size=population_size,
                       iterations=properties["iterations"])
    elif kind == "Hill":
        return HillClimber(selector,
                           amount_to_generate=population_size)
    elif kind == "Random":
        return RandomSearch(selector,
                            amount_to_generate=population_size)

class TestCase(Enum):
    COUNT_IDEALS = auto()
    CHECK_DISTRIBUTION = auto()

def decode_test(properties: str):
    test_string = properties["test"]
    if test_string == "count_ideals":
        return TestCase.COUNT_IDEALS
    elif test_string == "check_distribution":
        return TestCase.CHECK_DISTRIBUTION
