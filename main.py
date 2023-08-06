import random

import numpy as np

import HotEncoding
import SearchSpace
import utils
from BenchmarkProblems import CheckerBoard, OneMax, BinVal, TrapK, BT

import Version_B.FeatureDiscoverer
from Version_B import VariateModels
from Version_B.Sampler import ESTEEM_Sampler
from Version_B.SurrogateScorer import SurrogateScorer
import Version_B.FeatureExplorer

trap5 = TrapK.TrapK(5, 2)
checkerboard = CheckerBoard.CheckerBoardProblem(4, 4)
onemax = OneMax.OneMaxProblem(3)
binval = BinVal.BinValProblem(12, 2)
BT = BT.BTProblem(20, 3)

merging_power = 5


def get_problem_training_data(problem, sample_size):
    search_space = problem.get_search_space()
    random_candidates = [search_space.get_random_candidate() for _ in range(6000)]

    scores = [problem.score_of_candidate(c) for c in random_candidates]
    return (random_candidates, scores)


def pretty_print_features(problem, input_list_of_features, with_scores = False, combinatorial = False):
    hot_encoder = HotEncoding.HotEncoder(problem.get_search_space())
    def print_feature_only(feature):
        featureC = feature if combinatorial else hot_encoder.feature_from_hot_encoding(feature)
        problem.pretty_print_feature(featureC)

    def print_with_or_without_score(maybe_pair):
        if with_scores:
            feature, score = maybe_pair
            print_feature_only(feature)
            print(f"(has score {score:.2f})")
        else:
            print_feature_only(maybe_pair)

    for maybe_pair in input_list_of_features:
        print_with_or_without_score(maybe_pair)


def get_explainable_features(problem, training_data):
    print(f"The problem is {problem}")

    search_space = problem.get_search_space()
    (training_candidates, training_scores) = training_data

    # parameters
    importance_of_explainability = 0.5
    complexity_damping = 1

    feature_discoverer = Version_B.FeatureDiscoverer. \
        FeatureDiscoverer(search_space=search_space, candidateC_population=training_candidates,
                          fitness_scores=training_scores, merging_power=merging_power,
                          complexity_function=problem.get_complexity_of_feature,
                          complexity_damping=complexity_damping,
                          importance_of_explainability=importance_of_explainability)

    print("Exploring features...")
    feature_discoverer.generate_explainable_features()
    print("Obtaining the fit and unfit features")
    (fit_features, unfit_features) = feature_discoverer.get_explainable_features(criteria='fitness')
    (pop_features, unpop_features) = feature_discoverer.get_explainable_features(criteria='popularity')


    def remove_scores(list_of_features_with_scores):
        return utils.unzip(list_of_features_with_scores)[0]

    fit_features = remove_scores(fit_features)
    unfit_features = remove_scores(unfit_features)
    pop_features = remove_scores(pop_features)
    unpop_features = remove_scores(unpop_features)

    def debug_print_list_of_feature(criteria, feature_list):
        print(f"The features selected using the {criteria} are")
        hot_encoder = HotEncoding.HotEncoder(search_space)
        for featureH in feature_list:
            featureC = hot_encoder.feature_from_hot_encoding(featureH)
            problem.pretty_print_feature(featureC)
            print()

    debug_print_list_of_feature("fit", fit_features)
    debug_print_list_of_feature("unfit", unfit_features)
    debug_print_list_of_feature("pop", pop_features)
    debug_print_list_of_feature("unpop", unpop_features)

    return (fit_features, unfit_features, pop_features, unpop_features)

def test_surrogate_scorer(problem):
    search_space = problem.get_search_space()
    training_data = get_problem_training_data(problem, 1000)
    (fit_features, unfit_features, pop_features, unpop_features) = get_explainable_features(problem, training_data)

    def select_features_from_group_with_scores(features_with_scores, how_many):
        return utils.unzip(features_with_scores)[0][:how_many]

    to_keep_per_criteria = 30

    selected_fit_features = select_features_from_group_with_scores(fit_features, to_keep_per_criteria)
    selected_unfit_features = select_features_from_group_with_scores(unfit_features, to_keep_per_criteria)
    selected_pop_features = select_features_from_group_with_scores(pop_features, to_keep_per_criteria)

    to_keep_for_overall_selection = 50 // 3
    selected_features = utils.concat(group[:to_keep_for_overall_selection] for group
                                     in [selected_fit_features, selected_unfit_features, selected_pop_features])

    print("The selected features are:")
    pretty_print_features(problem, selected_features)

    print("Instantiating the surrogate scorer")
    trad_scorer = SurrogateScorer(model_power=2,
                             search_space=search_space,
                             featuresH=selected_features)
    print("And now we train the model")

    (training_candidates, training_scores) = training_data
    trad_scorer.train(training_candidates, training_scores)
    # scorer.make_picky()

    print(f"The model is now {trad_scorer}")


    print("We also train some other models")
    fit_scorer = SurrogateScorer(model_power=2,
                                  search_space=search_space,
                                  featuresH=selected_fit_features)
    unfit_scorer = SurrogateScorer(model_power=2,
                                 search_space=search_space,
                                 featuresH=selected_unfit_features)

    pop_scorer = SurrogateScorer(model_power=2,
                                 search_space=search_space,
                                 featuresH=selected_pop_features)

    fit_scorer.train(training_candidates, training_scores)
    unfit_scorer.train(training_candidates, training_scores)
    pop_scorer.train(training_candidates, training_scores)
    fit_scorer.set_deviation(kind='positive')
    unfit_scorer.set_deviation(kind='negative')

    def get_deviated_score(candidateC, based_on_trust=False):
        neutral_score = pop_scorer.get_surrogate_score_of_fitness(candidateC, based_on_trust)
        positive_score = fit_scorer.get_surrogate_score_of_fitness(candidateC, based_on_trust)
        negative_score = unfit_scorer.get_surrogate_score_of_fitness(candidateC, based_on_trust)

        return neutral_score + positive_score - negative_score


    def sanity_check():
        test_candidate = search_space.get_random_candidate()
        test_score = problem.score_of_candidate(test_candidate)
        surrogate_score = trad_scorer.get_surrogate_score_of_fitness(test_candidate)
        surrogate_mistrustful_score = trad_scorer.get_surrogate_score_of_fitness(test_candidate, based_on_trust=True)

        print(f"For a randomly generated candidate with actual score {test_score}, the surrogate score is {surrogate_score}")

    def print_data_for_analysis():
        (test_candidates, test_scores) = get_problem_training_data(problem, 1000)

        for (test_candidate, test_score) in zip(test_candidates, test_scores):
            surrogate_score = trad_scorer.get_surrogate_score_of_fitness(test_candidate)
            surrogate_mistrustful_score = trad_scorer.get_surrogate_score_of_fitness(test_candidate, based_on_trust=True)

            deviation_score = get_deviated_score(test_candidate, based_on_trust=False)
            deviation_score_mistrustful = get_deviated_score(test_candidate, based_on_trust=True)
            print(f"{test_score}"
                  f"\t{surrogate_score}"
                  f"\t{surrogate_mistrustful_score}"
                  f"\t{deviation_score}"
                  f"\t{deviation_score_mistrustful}")



    sanity_check()
    print_data_for_analysis()

def test_sampler(problem):
    search_space = problem.get_search_space()
    training_data = get_problem_training_data(problem, 120)
    (fit_features, unfit_features, pop_features, unpop_features) = get_explainable_features(problem, training_data)

    sampler = ESTEEM_Sampler(search_space, fit_features, unfit_features, unpop_features, importance_of_novelty=0.3)
    sampler.train(training_data[0], training_data[1])

    for _ in range(10):
        new_candidate = sampler.sample()
        actual_score = problem.score_of_candidate(new_candidate)

        problem.pretty_print_feature(new_candidate)
        print(f"Has score {actual_score}\n")

def test_explorer(problem):
    search_space = problem.get_search_space()
    print("Testing the explainable feature explorer")
    print(f"The problem is {problem}")

    print("Constructing the explorer")
    explorer = Version_B.FeatureExplorer.FeatureExplorer(search_space, merging_power, problem.get_complexity_of_feature,
                                                         importance_of_explainability = 0.5)

    training_samples, training_scores = get_problem_training_data(problem, 200)

    print("Then we obtain the meaningful features")
    (fit_features_and_scores), (unfit_features_and_scores), (popular_features_and_scores), (unpopular_features_and_scores) = \
        explorer.get_important_explainable_features(training_samples, training_scores)

    print("The fit features are")
    pretty_print_features(problem, fit_features_and_scores, combinatorial=True, with_scores=True)

    print("\n\nThe unfit features are")
    pretty_print_features(problem, unfit_features_and_scores, combinatorial=True, with_scores=True)

    print("\n\nThe popular features are")
    pretty_print_features(problem, popular_features_and_scores, combinatorial=True, with_scores=True)

    print("\n\nThe unpopular features are")
    pretty_print_features(problem, unpopular_features_and_scores, combinatorial=True, with_scores=True)





if __name__ == '__main__':
    test_explorer(trap5)

# TODO
# investigate why the scores are so bad
        # is it because not enough features are used?
        # test by changing the arrangement of the cells in binval
