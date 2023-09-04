import utils
from BenchmarkProblems import CombinatorialProblem, CheckerBoard, OneMax, BinVal, TrapK, BT, GraphColouring, Knapsack
from Version_C.Sampler import Sampler
from Version_C.FeatureFinder import ScoringCriterion, PopulationSamplePrecomputedData
import HotEncoding
import SearchSpace
from Version_C.FeatureFinder import find_features
from BenchmarkProblems.Knapsack import KnapsackConstraint

trap5 = TrapK.TrapK(5, 2)
checkerboard = CheckerBoard.CheckerBoardProblem(6, 6)
onemax = OneMax.OneMaxProblem(12)
binval = BinVal.BinValProblem(12, 2)
simpleBT = BT.SimplifiedBTProblem(25, 3)
almostBT = BT.BTProblem(30, 4, 28)
constrainedBT = BT.ExpandedBTProblem(almostBT, [BT.BTPredicate.EXCEEDS_WEEKLY_HOURS,
                                                BT.BTPredicate.BAD_MONDAY,
                                                BT.BTPredicate.BAD_TUESDAY,
                                                BT.BTPredicate.BAD_WEDNESDAY,
                                                BT.BTPredicate.BAD_THURSDAY,
                                                BT.BTPredicate.BAD_FRIDAY,
                                                BT.BTPredicate.BAD_SATURDAY,
                                                BT.BTPredicate.BAD_SUNDAY])

graph_colouring = GraphColouring.GraphColouringProblem(4, 10, 0.5)
knapsack = Knapsack.KnapsackProblem(50.00, 1000, 15)
c_knapsack = Knapsack.ConstrainedKnapsackProblem(knapsack, [KnapsackConstraint.BEACH, KnapsackConstraint.FLYING,
                                                            KnapsackConstraint.WITHIN_WEIGHT,
                                                            KnapsackConstraint.WITHIN_VOLUME])

guaranteed_depth = 5
extra_depth = 5
importance_of_explainability = 0.5


def get_problem_training_data(problem: CombinatorialProblem.CombinatorialProblem,
                              sample_size) -> (list[SearchSpace.Candidate], list[float]):
    random_candidates = [problem.get_random_candidate_solution() for _ in range(sample_size)]

    scores = [problem.score_of_candidate(c) for c in random_candidates]
    return random_candidates, scores


def get_problem_compact_training_data(problem: CombinatorialProblem.CombinatorialProblem,
                                      sample_size) -> PopulationSamplePrecomputedData:
    training_samples, fitness_list = get_problem_training_data(problem, sample_size)
    # print("The generated samples are:")
    # for sample, fitness in zip(training_samples, fitness_list):
    #     print(f"{problem.candidate_repr(sample)}\n(has score {fitness:.2f})")
    return PopulationSamplePrecomputedData(problem.search_space, training_samples, fitness_list)


def pretty_print_features(problem: CombinatorialProblem.CombinatorialProblem, input_list_of_features, with_scores=False,
                          combinatorial=False):
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
                 criterion: ScoringCriterion,
                 amount_requested: int):
    print("Finding the features...")
    features, scores = find_features(problem=problem,
                                     sample_data=sample_data,
                                     criterion=criterion,
                                     importance_of_explainability=importance_of_explainability,
                                     guaranteed_depth=guaranteed_depth,
                                     extra_depth=extra_depth,
                                     amount_requested=amount_requested)
    return features


def get_sampler(problem: CombinatorialProblem,
                training_data: PopulationSamplePrecomputedData,
                amount_of_features_per_sampler,
                is_maximisation_task=True) -> Sampler:
    print("Constructing the sampler, involves finding:")
    print("\t -fit features")
    fit_features = get_features(problem, training_data, ScoringCriterion.HIGH_FITNESS, amount_of_features_per_sampler)
    print("\t -unfit features")
    unfit_features = get_features(problem, training_data, ScoringCriterion.LOW_FITNESS, amount_of_features_per_sampler)
    print("\t -novel features")
    novel_features = get_features(problem, training_data, ScoringCriterion.NOVELTY, amount_of_features_per_sampler)

    wanted_features, unwanted_features = (fit_features, unfit_features) if is_maximisation_task else (
    unfit_features, fit_features)

    sampler = Sampler(search_space=problem.search_space,
                      wanted_features=wanted_features,
                      unwanted_features=unwanted_features,
                      unpopular_features=novel_features,
                      importance_of_novelty=0.1)

    print("Then we train the sampler")
    sampler.train(training_data)
    return sampler


def get_good_samples(sampler, problem, attempts, keep, maximise=True):
    samples = [sampler.sample() for _ in range(attempts)]
    samples_with_scores = [(sample, problem.score_of_candidate(sample)) for sample in samples]
    samples_with_scores.sort(key=utils.second, reverse=maximise)
    return utils.unzip(samples_with_scores[:keep])


if __name__ == '__main__':
    problem = trap5
    maximise = True
    training_data = get_problem_compact_training_data(problem, sample_size=1200)
    print(f"The problem is {problem}")
    print("More specifically, it is")
    print(problem.long_repr())
    criteria = ScoringCriterion.HIGH_FITNESS if maximise else ScoringCriterion.LOW_FITNESS
    requested_amount_of_features = 12
    features = get_features(problem, training_data, criteria, requested_amount_of_features)

    print(f"For the problem {problem}, the found features are:")
    pretty_print_features(problem, features, combinatorial=True)

    # sampler = get_sampler(problem, training_data, requested_amount_of_features // 2, maximise)

    # print("We can sample some individuals")
    # good_samples, good_sample_scores = get_good_samples(sampler, problem, 30, 6, maximise)
    # for good_sample, good_score in zip(good_samples, good_sample_scores):
    #     print(f"{problem.candidate_repr(good_sample)}\n(Has score {good_score:.2f})\n")
