from BenchmarkProblems import CombinatorialProblem, CheckerBoard, OneMax, BinVal, TrapK, BT, GraphColouring, Knapsack
from BenchmarkProblems.ArtificialProblem import ArtificialProblem
import SearchSpace
from BenchmarkProblems.Knapsack import KnapsackConstraint
from Version_E.MeasurableCriterion.CriterionUtilities import All, Any, Not, Balance
from Version_E.MeasurableCriterion.GoodFitness import HighFitness, ConsistentFitness
from Version_E.MeasurableCriterion.Explainability import Explainability
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.InterestingAlgorithms.Miner import FeatureSelector
from Version_E.InterestingAlgorithms.ConstructiveMiner import ConstructiveMiner
from Version_E.InterestingAlgorithms.DestructiveMiner import DestructiveMiner

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


def pretty_print_features(problem: CombinatorialProblem.CombinatorialProblem, input_list_of_features, with_scores=False):
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


if __name__ == '__main__':

    problem = artificial_problem
    is_explainable = Explainability(problem)
    has_high_fitness_consistently = Balance([HighFitness(), ConsistentFitness()], weights=[0, 6])

    criterion = Balance([is_explainable, has_high_fitness_consistently], weights=[1, 2])

    training_data = get_training_data(problem, sample_size=3000)
    print(f"The problem is {problem}")
    print("More specifically, it is")
    print(problem.long_repr())

    selector = FeatureSelector(training_data, criterion)

    def get_miner(kind: str, stochastic: bool, population_size: int):
        if kind == "Constructive":
            return ConstructiveMiner(selector, population_size,
                                     stochastic=stochastic,
                                     at_most_parameters=5)
        elif kind == "Destructive":
            return DestructiveMiner(selector, population_size,
                                    stochastic,
                                    at_least_parameters=1)

    miners = [get_miner(kind, stochastic, population_size)
              for kind in ["Destructive", "Constructive"]
              for stochastic in [True, False]
              for population_size in [30, 50]]

    for miner in miners:
        features = miner.get_meaningful_features(12)
        print("features_found:")
        pretty_print_features(problem, features)

