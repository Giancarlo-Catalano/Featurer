from BenchmarkProblems import CombinatorialProblem, CheckerBoard, OneMax, BinVal, TrapK, BT, GraphColouring, Knapsack
from BenchmarkProblems.ArtificialProblem import ArtificialProblem
import HotEncoding
import SearchSpace
from BenchmarkProblems.Knapsack import KnapsackConstraint
from Version_E import MeasurableCriterion
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.MinerUtilities import FeatureSelector, ConstructiveMiner, DestructiveMiner

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
    # print("The generated samples are:")
    # for sample, fitness in zip(training_samples, fitness_list):
    #     print(f"{problem.candidate_repr(sample)}\n(has score {fitness:.2f})")
    return PrecomputedPopulationInformation(problem.search_space, training_samples, fitness_list)


def pretty_print_features(problem: CombinatorialProblem.CombinatorialProblem, input_list_of_features, with_scores=False,
                          combinatorial=True):
    """prints the passed features, following the structure specified by the problem"""
    hot_encoder = HotEncoding.HotEncoder(problem.search_space)

    def print_feature_only(feature):
        featureC = feature if combinatorial else hot_encoder.feature_from_hot_encoding(feature)
        print(f"{problem.feature_repr(featureC)}")

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

    problem = trap5
    criteria_and_weights = [(MeasurableCriterion.explainability_of(problem), 5),
                            (MeasurableCriterion.MeanFitnessCriterion(), 5),
                            (MeasurableCriterion.FitnessConsistencyCriterion(), 2)]

    training_data = get_training_data(problem, sample_size=1200)
    print(f"The problem is {problem}")
    print("More specifically, it is")
    print(problem.long_repr())

    selector = FeatureSelector(training_data, criteria_and_weights)
    amount_to_keep_in_each_layer = 120


    def get_miner(kind: str, stochastic: bool):
        if kind == "Constructive":
            return ConstructiveMiner(selector, amount_to_keep_in_each_layer,
                                     stochastic=stochastic,
                                     at_most_parameters=5)
        elif kind == "Destructive":
            return DestructiveMiner(selector, amount_to_keep_in_each_layer,
                                    stochastic,
                                    at_least_parameters=1)


    cs_miner = get_miner("Constructive", stochastic=True)
    ch_miner = get_miner("Constructive", stochastic=False)
    ds_miner = get_miner("Destructive", stochastic=True)
    dh_miner = get_miner("Destructive", stochastic=False)

    miners = [cs_miner, ch_miner, ds_miner, dh_miner]
    for miner in miners:
        features = miner.mine_features()
        features = [feature.to_legacy_feature() for feature in features]
        print("features_found:")
        pretty_print_features(problem, features)

    """features = get_features_version_D(training_data, criteria_and_weights,
                                      guaranteed_depth=1,
                                      explored_depth=5)
    """
