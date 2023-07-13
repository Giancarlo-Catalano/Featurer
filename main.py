import ProgressiveFeatures
import utils
from BenchmarkProblems import CheckerBoard, OneMax, BinVal, TrapK
import AutomatedFeatureDetection
import numpy as np
import warnings


def test_problem(problem):
    print(f"The problem is {problem}")
    features_with_scores = AutomatedFeatureDetection.induce_features(problem.get_search_space(),
                                                                     problem.score_of_candidate,
                                                                     problem.get_complexity_of_feature,
                                                                     importance_of_explainability=0.5,
                                                                     merging_power=2)

    for (feature, score) in features_with_scores:
        print(f"Feature:")
        problem.pretty_print_feature(feature)
        print(f"(has score {score:.2f})")


def test_problem_thoroughly(problem):
    print(f"The problem is {problem}")
    AutomatedFeatureDetection.induce_features_to_generate_solutions(problem.get_search_space(),
                                                                    problem.score_of_candidate,
                                                                    problem.get_complexity_of_feature,
                                                                    problem.pretty_print_feature,
                                                                    importance_of_explainability=0.4,
                                                                    merging_power = 2)


trap5 = TrapK.TrapK(4, 3)
checkerboard = CheckerBoard.CheckerBoardProblem(3, 3)
onemax = OneMax.OneMaxProblem(6)
binval = BinVal.BinValProblem(6, 2)

def describe_problem():
    test_problem_thoroughly(checkerboard)

if __name__ == '__main__':
    utils.stop_for_every_warning(describe_problem)

