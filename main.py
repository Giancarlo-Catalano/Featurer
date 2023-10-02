import sys

from BenchmarkProblems import CombinatorialProblem, CheckerBoard, OneMax, BinVal, TrapK, BT, GraphColouring, Knapsack
from BenchmarkProblems.ArtificialProblem import ArtificialProblem
import SearchSpace
from BenchmarkProblems.Knapsack import KnapsackConstraint
from Version_E.MeasurableCriterion.CriterionUtilities import Any, Not, Balance
from Version_E.MeasurableCriterion.GoodFitness import HighFitness, ConsistentFitness
from Version_E.MeasurableCriterion.Explainability import Explainability
from Version_E.MeasurableCriterion.Robustness import Robustness
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.InterestingAlgorithms.Miner import FeatureSelector
from Version_E.InterestingAlgorithms.ConstructiveMiner import ConstructiveMiner
from Version_E.InterestingAlgorithms.DestructiveMiner import DestructiveMiner
from Version_E.Testing import TestingUtilities, Problems, Criteria

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

graph_colouring = GraphColouring.GraphColouringProblem(3, 10, 0.5)
knapsack = Knapsack.KnapsackProblem(50.00, 1000, 15)
constrained_knapsack = Knapsack.ConstrainedKnapsackProblem(knapsack,
                                                           [KnapsackConstraint.BEACH, KnapsackConstraint.FLYING,
                                                            KnapsackConstraint.WITHIN_WEIGHT])
artificial_problem = ArtificialProblem(12, 3, 4, False)


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


def test_command_line():
    command_line_arguments = sys.argv
    if len(command_line_arguments) < 2:
        raise Exception("Not enough arguments")

    # settings = TestingUtilities.to_json_object(command_line_arguments[1])

    settings = dict()
    settings["problem"] = Problems.make_problem("trapk", "medium")
    settings["criterion"] = Criteria.high_fitness_and_explainable
    settings["test"] = {"which": "count_ideals",
                        "runs": 12}
    settings["miner"] = {"which": "destructive",
                         "stochastic": False,
                         "at_least": 1,
                         "population_size": 60}
    settings["sample_size"] = 1200
    TestingUtilities.run_test(settings)


def test_miner():
    problem = artificial_problem
    is_explainable = Explainability(problem)
    has_good_fitness_consistently = Balance([(HighFitness()), ConsistentFitness()], weights=[2, 1])
    robust_to_changes = Balance([Robustness(0, 1),
                                 Robustness(1, 2),
                                 Robustness(2, 5)],
                                weights=[4, 2, 1])

    criterion = Balance([is_explainable, has_good_fitness_consistently, robust_to_changes],
                        weights=[1, 1, 0])

    training_data = get_training_data(problem, sample_size=3000)
    print(f"The problem is {problem}")
    print("More specifically, it is")
    print(problem.long_repr())

    selector = FeatureSelector(training_data, criterion)

    miner = DestructiveMiner(selector,
                             amount_to_keep_in_each_layer=72,
                             stochastic=False,
                             at_least_parameters=1)

    features = miner.get_meaningful_features(12, cull_subsets=True)
    print("features_found:")
    for feature in features:
        print(problem.feature_repr(feature.to_legacy_feature()))
        print(criterion.describe_feature(feature, training_data))
        print("\n")


if __name__ == '__main__':
    test_miner()