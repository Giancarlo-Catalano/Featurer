#!/usr/bin/env python3
import sys

import utils
from BenchmarkProblems import CombinatorialProblem, CheckerBoard, OneMax, BinVal, TrapK, BT, GraphColouring, Knapsack, \
    FourPeaks
from BenchmarkProblems.ArtificialProblem import ArtificialProblem
import SearchSpace
from BenchmarkProblems.Knapsack import KnapsackConstraint
from Version_E.Feature import Feature
from Version_E.MeasurableCriterion.CriterionUtilities import Any, Not, Balance
from Version_E.MeasurableCriterion.GoodFitness import HighFitness, ConsistentFitness, FitnessHigherThanAverage
from Version_E.MeasurableCriterion.Explainability import Explainability
from Version_E.MeasurableCriterion.Robustness import Robustness
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.InterestingAlgorithms.Miner import FeatureSelector
from Version_E.InterestingAlgorithms.ConstructiveMiner import ConstructiveMiner
from Version_E.InterestingAlgorithms.DestructiveMiner import DestructiveMiner
from Version_E.Testing import TestingUtilities, Problems, Criteria
from Version_E.Testing.Tests import make_csv_for_connectedness


def get_random_candidates_and_fitnesses(problem: CombinatorialProblem.CombinatorialProblem,
                                        sample_size) -> (list[SearchSpace.Candidate], list[float]):
    random_candidates = [problem.get_random_candidate_solution() for _ in range(sample_size)]

    scores = [problem.score_of_candidate(c) for c in random_candidates]
    return random_candidates, scores


def get_training_data(problem: CombinatorialProblem.CombinatorialProblem,
                      sample_size) -> PrecomputedPopulationInformation:
    training_samples, fitness_list = get_random_candidates_and_fitnesses(problem, sample_size)
    return PrecomputedPopulationInformation(problem.search_space, training_samples, fitness_list)


def test_command_line():
    # command_line_arguments = sys.argv
    # if len(command_line_arguments) < 2:
    #    raise Exception("Not enough arguments")

    # settings = TestingUtilities.to_json_object(command_line_arguments[1])



    settings = dict()

    settings["problem"] = {"which": "graph",
                           "amount_of_colours":4,
                           "amount_of_nodes": 6,
                           "chance_of_connection": 0.3}

    settings["criterion"] = {"which": "balance",
                             "arguments": [{"which": "high_fitness"},
                                           {"which": "explainability"}],
                             "weights": [2, 1]}


    settings["test"] = {"which": "check_connectedness",
                        "features_per_run": 200,
                        "runs": 24}
    settings["miner"] = {"which": "constructive",
                         "stochastic": False,
                         "at_most": 5,
                         "population_size": 36}
    settings["sample_size"] = 2400
    TestingUtilities.run_test(settings)


if __name__ == '__main__':
    #make_csv_for_connectedness("check_connectedness~graph~constructive_(10-10)_[15_42].json", "conn_out_3.csv")
    test_command_line()
