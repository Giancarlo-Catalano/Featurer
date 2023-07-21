import HotEncoding
import SearchSpace
import utils
from BenchmarkProblems import CheckerBoard, OneMax, BinVal, TrapK

import Version_B.FeatureDiscoverer

trap5 = TrapK.TrapK(5, 3)
checkerboard = CheckerBoard.CheckerBoardProblem(6, 6)
onemax = OneMax.OneMaxProblem(12)
binval = BinVal.BinValProblem(12, 2)


def test_FeatureDiscoverer(problem):
    print(f"The problem is {problem}")


    ss = problem.get_search_space()
    random_candidates = [ss.get_random_candidate() for _ in range(6000)]


    scores = [problem.score_of_candidate(c) for c in random_candidates]
    importance_of_explainability = 0.5
    complexity_damping = 2
    merging_power = 2

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
            featuresH.sort(key=utils.second, reverse=True)
            for featureH, score in featuresH:
                print("-" * 10)
                problem.pretty_print_feature(fd.hot_encoder.feature_from_hot_encoding(featureH))
                print(f"(has score {score:.2f})")
                print("-" * 10)

    # print(f"The trivial hot features are:")
    # print_list_of_features(trivial_featuresH)

    print("Exploring features...")
    fd.generate_explainable_features()


    print("Obtaining the good and bad features")
    (good_features, bad_features) = fd.get_important_features(on_commonality=False)
    (popular_features, unpopular_features)    = fd.get_important_features(on_commonality=True)


    print("The good features are")
    print_list_of_features(good_features, with_scores=True)
    print("The bad features are")
    print_list_of_features(bad_features, with_scores=True)
    print("The popular features are")
    print_list_of_features(popular_features, with_scores=True)
    print("The unpopular features are")
    print_list_of_features(unpopular_features, with_scores=True)




if __name__ == '__main__':
    test_FeatureDiscoverer(checkerboard)

