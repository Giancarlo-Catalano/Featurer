import utils
from BenchmarkProblems import CheckerBoard, OneMax, BinVal, TrapK

import Version_B.FeatureDiscoverer

trap5 = TrapK.TrapK(4, 3)
checkerboard = CheckerBoard.CheckerBoardProblem(3, 3)
onemax = OneMax.OneMaxProblem(6)
binval = BinVal.BinValProblem(6, 2)


def test_FeatureDiscoverer(problem):
    print(f"The problem is {problem}")


    ss = problem.get_search_space()
    random_candidates = [ss.get_random_candidate() for _ in range(6000)]
    scores = [problem.score_of_candidate(c) for c in random_candidates]
    importance_of_explainability = 0.30
    complexity_damping = 2
    merging_power = 4

    fd = Version_B.FeatureDiscoverer.FeatureDiscoverer(search_space=ss,
                                                       candidateC_population=random_candidates,
                                                       fitness_scores=scores,
                                                       merging_power=merging_power,
                                                       complexity_function=problem.get_complexity_of_feature,
                                                       complexity_damping=complexity_damping,
                                                       importance_of_explainability=importance_of_explainability)
    def print_list_of_features(featuresH, with_scores = False):
        if not with_scores:
            for featureH in featuresH:
                print("-"*10)
                problem.pretty_print_feature(fd.hot_encoder.feature_from_hot_encoding(featureH))
                print("-"*10)
        else:
            for featureH, score in featuresH:
                print("-" * 10)
                problem.pretty_print_feature(fd.hot_encoder.feature_from_hot_encoding(featureH))
                print(f"(has score {score:.2f})")
                print("-" * 10)

    trivial_featuresH = fd.get_initial_features()

    print(f"The trivial hot features are:")
    print_list_of_features(trivial_featuresH)

    print("Starting to generate the next generation")
    new_features = fd.get_next_wave_of_features(trivial_featuresH, trivial_featuresH)
    print("The generated features are:")
    print_list_of_features(new_features, with_scores=True)

    # print("And if we try to generate yet new ones... we should get lots of duplicates...")
    # redundant_features = fd.get_next_wave_of_features(new_features, trivial_featuresH)
    # print_list_of_features(redundant_features)




if __name__ == '__main__':
    test_FeatureDiscoverer(checkerboard)

