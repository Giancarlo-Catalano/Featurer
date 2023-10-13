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

trap5 = TrapK.TrapK(5, 3)
checkerboard = CheckerBoard.CheckerBoardProblem(5, 5)
onemax = OneMax.OneMaxProblem(12)
binval = BinVal.BinValProblem(12, 2)
almostBT = BT.BTProblem(12, 4, 56)
constrained_BT = BT.ExpandedBTProblem(almostBT, [BT.BTPredicate.EXCEEDS_WEEKLY_HOURS,
                                                 BT.BTPredicate.BAD_MONDAY,
                                                 BT.BTPredicate.BAD_TUESDAY,
                                                 BT.BTPredicate.BAD_WEDNESDAY,
                                                 BT.BTPredicate.BAD_THURSDAY,
                                                 BT.BTPredicate.BAD_FRIDAY,
                                                 BT.BTPredicate.BAD_SATURDAY,
                                                 BT.BTPredicate.BAD_SUNDAY])

graph_colouring = GraphColouring.GraphColouringProblem(3, 6, 0.5)
knapsack = Knapsack.KnapsackProblem(50.00, 1000, 15)
constrained_knapsack = Knapsack.ConstrainedKnapsackProblem(knapsack,
                                                           [KnapsackConstraint.BEACH, KnapsackConstraint.FLYING,
                                                            KnapsackConstraint.WITHIN_WEIGHT])
artificial_problem = ArtificialProblem(12, 3, 4, True)
four_peaks = FourPeaks.FourPeaksProblem(12, 4)


def get_random_candidates_and_fitnesses(problem: CombinatorialProblem.CombinatorialProblem,
                                        sample_size) -> (list[SearchSpace.Candidate], list[float]):
    random_candidates = [problem.get_random_candidate_solution() for _ in range(sample_size)]

    scores = [problem.score_of_candidate(c) for c in random_candidates]
    return random_candidates, scores


def get_training_data(problem: CombinatorialProblem.CombinatorialProblem,
                      sample_size) -> PrecomputedPopulationInformation:
    training_samples, fitness_list = get_random_candidates_and_fitnesses(problem, sample_size)
    return PrecomputedPopulationInformation(problem.search_space, training_samples, fitness_list)


def pretty_print_features(problem: CombinatorialProblem.CombinatorialProblem, input_list_of_features,
                          with_scores=False):
    """prints the passed features, following the structure specified by the problem"""

    def print_feature_only(feature):
        print(f"{problem.feature_repr(feature)}")

    def print_with_or_without_score(maybe_pair):
        if with_scores:
            feature, score = maybe_pair
            print_feature_only(feature)
            print(f"(has score {score:.2f})")
        else:
            print_feature_only(maybe_pair)

    for maybe_pair in input_list_of_features:
        print_with_or_without_score(maybe_pair)
        print("\n")


def show_all_ideals():
    problems_with_ideals = [onemax, binval, trap5, artificial_problem, checkerboard]

    for problem in problems_with_ideals:
        print(f"The problem is {problem}, more specifically \n{problem.long_repr()}")
        print("\n The ideals are ")
        for feature in problem.get_ideal_features():
            print(f"\n{problem.feature_repr(feature)}")

        print("_" * 40)



constructive_miners = [{"which": "constructive",
                        "stochastic": stochastic_item,
                        "at_most": 5,
                        "population_size": population_item}
                       for stochastic_item in [True, False]
                       for population_item in [36, 72, 144]]

destructive_miners = [{"which": "destructive",
                        "stochastic": stochastic_item,
                        "at_least": 1,
                        "population_size": population_item}
                       for stochastic_item in [True, False]
                       for population_item in [36, 72, 144]]

ga_miners = [{"which": "ga",
             "iterations": iteration_item,
             "population_size": population_item}
            for iteration_item in [5, 10, 20]
            for population_item in [36, 72, 144]]


random_miners = [{"which": "random",
                  "population_size": population_item}
                 for population_item in [36, 72, 144]]

hill_climber = [{"which": "hill_climber",
                 "population_size": population_item}
                for population_item in [36, 72, 144]]

many_miners = utils.concat_lists([constructive_miners, destructive_miners, ga_miners, random_miners, hill_climber])


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


def test_miner():
    problem = artificial_problem
    is_explainable = Explainability(problem)
    has_good_fitness_consistently = Balance([FitnessHigherThanAverage(), ConsistentFitness()], weights=[2, 1])
    robust_to_changes = Balance([Robustness(0, 1),
                                 Robustness(1, 2),
                                 Robustness(2, 5)],
                                weights=[4, 2, 1])

    criterion = Balance([is_explainable, has_good_fitness_consistently],
                        weights=[2, 1])

    training_data = get_training_data(problem, sample_size=1200)
    print(f"The problem is {problem}")
    print("More specifically, it is")
    print(problem.long_repr())

    selector = FeatureSelector(training_data, criterion)

    miner = DestructiveMiner(selector,
                             amount_to_keep_in_each_layer=144,
                             stochastic=False,
                             at_least_parameters=1)

    features = miner.get_meaningful_features(12, cull_subsets=True)

    # debug
    """relevant_features = [Feature.from_legacy_feature(ideal, problem.search_space)
                         for ideal in problem.important_features]
    print("The scores of the intended features are:")
    for relevant_feature in relevant_features:
        print(f"For feature {relevant_feature}, the description is")
        print(miner.feature_selector.criterion.describe_feature(relevant_feature, miner.feature_selector.ppi))
    """
    # end debug
    print("features_found:")
    for feature in features:
        print(problem.feature_repr(feature.to_legacy_feature()))
        print(criterion.describe_feature(feature, training_data))
        print("\n")


if __name__ == '__main__':
    #make_csv_for_connectedness("check_connectedness~graph~constructive_(10-10)_[15_42].json", "conn_out_3.csv")
    test_command_line()
