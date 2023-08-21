from BenchmarkProblems import CombinatorialProblem, CheckerBoard, OneMax, BinVal, TrapK, BT, GraphColouring, Knapsack
from Version_B.Sampler import Sampler
from Version_B.FeatureFinder import ScoringCriterion, PopulationSamplePrecomputedData
import HotEncoding
import SearchSpace
from Version_B.FeatureFinder import find_features

trap5 = TrapK.TrapK(5, 3)
checkerboard = CheckerBoard.CheckerBoardProblem(4, 4)
onemax = OneMax.OneMaxProblem(12)
binval = BinVal.BinValProblem(12, 2)
BT = BT.BTProblem(25, 3)
graph_colouring = GraphColouring.GraphColouringProblem(3, 6, 0.5)
knapsack = Knapsack.KnapsackProblem(50.00, 1000, 10)

depth = 4
importance_of_explainability = 0.2


def get_problem_training_data(problem: CombinatorialProblem.CombinatorialProblem,
                              sample_size) -> (list[SearchSpace.Candidate], list[float]):
    random_candidates = [problem.get_random_candidate_solution() for _ in range(sample_size)]

    scores = [problem.score_of_candidate(c) for c in random_candidates]
    return random_candidates, scores


def get_problem_compact_training_data(problem: CombinatorialProblem.CombinatorialProblem,
                                      sample_size) -> PopulationSamplePrecomputedData:
    training_samples, fitness_list = get_problem_training_data(problem, sample_size)
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
                 criteria: ScoringCriterion,
                 amount_requested: int):
    print("Finding the features...")
    features, scores = find_features(problem=problem,
                                     depth=depth,
                                     importance_of_explainability=importance_of_explainability,
                                     heuristic=True,
                                     sample_data=sample_data,
                                     criteria=criteria,
                                     amount_requested=amount_requested)
    return features


def get_sampler(problem: CombinatorialProblem,
                training_data: PopulationSamplePrecomputedData,
                amount_of_features_per_sampler,
                is_maximisation_task = True) -> Sampler:
    print("Constructing the sampler, involves finding:")
    print("\t -fit features")
    fit_features = get_features(problem, training_data, ScoringCriterion.HIGH_FITNESS, amount_of_features_per_sampler)
    print("\t -unfit features")
    unfit_features = get_features(problem, training_data, ScoringCriterion.LOW_FITNESS, amount_of_features_per_sampler)
    print("\t -novel features")
    novel_features = get_features(problem, training_data, ScoringCriterion.NOVELTY, amount_of_features_per_sampler)

    wanted_features, unwanted_features = (fit_features, unfit_features) if is_maximisation_task else (unfit_features, fit_features)

    sampler = Sampler(search_space=problem.search_space,
                      wanted_features=wanted_features,
                      unwanted_features=unwanted_features,
                      unpopular_features=novel_features,
                      importance_of_novelty=0.1)

    print("Then we train the sampler")
    sampler.train(training_data)
    return sampler



if __name__ == '__main__':
    problem = knapsack
    maximise = False
    training_data = get_problem_compact_training_data(problem, sample_size=1000)
    print(f"The problem is {problem}")
    criteria = ScoringCriterion.HIGH_FITNESS if maximise else ScoringCriterion.LOW_FITNESS
    requested_amount_of_features = problem.search_space.total_cardinality
    features = get_features(problem, training_data, criteria, requested_amount_of_features)

    print(f"For the problem {problem}, the found features are:")
    pretty_print_features(problem, features, combinatorial=True)

    sampler = get_sampler(problem, training_data, requested_amount_of_features//2)

    print("We can sample some individuals")
    how_many_to_sample = 6
    for _ in range(how_many_to_sample):
        new_individual = sampler.sample()
        actual_score = problem.score_of_candidate(new_individual)

        print(f"{problem.candidate_repr(new_individual)}\n"
              f"Has score {actual_score}\n")



