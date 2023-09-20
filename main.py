import Version_D.MeasurableCriterion
import utils
from BenchmarkProblems import CombinatorialProblem, CheckerBoard, OneMax, BinVal, TrapK, BT, GraphColouring, Knapsack
from BenchmarkProblems.ArtificialProblem import ArtificialProblem
from Version_C.Sampler import Sampler
from Version_C.FeatureFinder import ScoringCriterion, PopulationSamplePrecomputedData
import HotEncoding
import SearchSpace
from Version_C.FeatureFinder import find_features
from BenchmarkProblems.Knapsack import KnapsackConstraint
from Version_D.Miner import Parameters, Miner
from Version_D.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_D import MeasurableCriterion

trap5 = TrapK.TrapK(5, 3)
checkerboard = CheckerBoard.CheckerBoardProblem(3, 3)
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


def get_features(problem: CombinatorialProblem,
                 sample_data: PopulationSamplePrecomputedData,
                 criteria_and_weights: [(ScoringCriterion, float)],
                 amount_requested=12,
                 guaranteed_depth=1,
                 explored_depth=6,
                 strategy="always heuristic",
                 search_multiplier=6):
    print("Finding the features...")
    features, scores = find_features(problem=problem,
                                     sample_data=sample_data,
                                     criteria_and_weights=criteria_and_weights,
                                     guaranteed_depth=guaranteed_depth,
                                     extra_depth=explored_depth,
                                     amount_requested=amount_requested,
                                     strategy=strategy,
                                     search_multiplier=search_multiplier)
    return features


def get_features_version_D(sample_data: PrecomputedPopulationInformation,
                           criteria_and_weights: [(ScoringCriterion, float)],
                           guaranteed_depth=1,
                           explored_depth=6):
    schedule = Parameters.get_parameter_schedule(search_space=sample_data.search_space,
                                                 guaranteed_depth=guaranteed_depth,
                                                 explored_depth=explored_depth,
                                                 search_method=Parameters.SearchMethod.STOCHASTIC_SEARCH,
                                                 criteria_and_weights=criteria_and_weights,
                                                 proportionality=Parameters.Proportionality.PROBLEM_PARAMETERS,
                                                 thoroughness=Parameters.Thoroughness.MOST,
                                                 criteria_start=Parameters.CriteriaStart.FROM_MIDPOINT)

    found_features, scores = Miner.mine_meaningful_features(sample_data, schedule)

    found_features = [feature.to_legacy_feature() for feature in found_features]
    return found_features[:12]


if __name__ == '__main__':
    problem = constrained_BT
    criteria_and_weights = [(MeasurableCriterion.explainability_of(problem), 5),
                            (MeasurableCriterion.MeanFitnessCriterion(), -5),
                            (MeasurableCriterion.FitnessConsistencyCriterion(), 2),
                            (MeasurableCriterion.CorrelationCriterion(), 5)]

    training_data = get_training_data(problem, sample_size=1200)
    print(f"The problem is {problem}")
    print("More specifically, it is")
    print(problem.long_repr())

    features = Miner.create_through_destruction(training_data, criteria_and_weights, 120)
    features = [feature.to_legacy_feature() for feature in features]


    """features = get_features_version_D(training_data, criteria_and_weights,
                                      guaranteed_depth=1,
                                      explored_depth=5)
    """
    print(f"For the problem {problem}, the found features with {criteria_and_weights = } are:")
    pretty_print_features(problem, features)
