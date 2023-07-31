import random

import numpy as np

import HotEncoding
import SearchSpace
import utils
from BenchmarkProblems import CheckerBoard, OneMax, BinVal, TrapK, BT

import Version_B.FeatureDiscoverer
from Version_B import VariateModels
from Version_B.SurrogateScorer import SurrogateScorer

trap5 = TrapK.TrapK(5, 2)
checkerboard = CheckerBoard.CheckerBoardProblem(4, 4)
onemax = OneMax.OneMaxProblem(12)
binval = BinVal.BinValProblem(12, 2)
BT = BT.BTProblem(20, 3)


def get_problem_training_data(problem, sample_size):
    search_space = problem.get_search_space()
    random_candidates = [search_space.get_random_candidate() for _ in range(6000)]

    scores = [problem.score_of_candidate(c) for c in random_candidates]
    return (random_candidates, scores)


def pretty_print_features(problem, features):
    hot_encoder = HotEncoding.HotEncoder(problem.get_search_space())
    for featureH in features:
        featureC = hot_encoder.feature_from_hot_encoding(featureH)
        problem.pretty_print_feature(featureC)
        print()

def test_FeatureDiscoverer(problem):
    print(f"The problem is {problem}")

    search_space = problem.get_search_space()
    random_candidates = [search_space.get_random_candidate() for _ in range(6000)]

    scores = [problem.score_of_candidate(c) for c in random_candidates]
    importance_of_explainability = 0.75
    complexity_damping = 1
    merging_power = 3

    fd = Version_B.FeatureDiscoverer.FeatureDiscoverer(search_space=search_space,
                                                       candidateC_population=random_candidates,
                                                       fitness_scores=scores,
                                                       merging_power=merging_power,
                                                       complexity_function=problem.get_complexity_of_feature,
                                                       complexity_damping=complexity_damping,
                                                       importance_of_explainability=importance_of_explainability)

    def print_list_of_features(featuresH, with_scores=False):
        to_show = 6
        if not with_scores:
            for featureH in featuresH[:to_show]:
                print("-" * 10)
                problem.pretty_print_feature(fd.hot_encoder.feature_from_hot_encoding(featureH))
                print("-" * 10)
        else:
            featuresH.sort(key=utils.second, reverse=True)
            for featureH, score in featuresH[:to_show]:
                print("-" * 10)
                problem.pretty_print_feature(fd.hot_encoder.feature_from_hot_encoding(featureH))
                print(f"(has score {score:.2f})")
                print("-" * 10)

    # print(f"The trivial hot features are:")
    # print_list_of_features(trivial_featuresH)

    print("Exploring features...")
    fd.generate_explainable_features()

    print("Obtaining the good and bad features")
    (fit_features, unfit_features) = fd.get_explainable_features(criteria='fitness')
    (popular_features, unpopular_features) = fd.get_explainable_features(criteria='popularity')

    print("The good features are")
    print_list_of_features(fit_features, with_scores=True)
    print("The bad features are")
    print_list_of_features(unfit_features, with_scores=True)
    print("The popular features are")
    print_list_of_features(popular_features, with_scores=True)
    print("The unpopular features are")
    print_list_of_features(unpopular_features, with_scores=True)

    # this section is for surrogate scoring
    hot_encoder = HotEncoding.HotEncoder(search_space)
    variate_models: VariateModels.VariateModels = VariateModels.VariateModels(search_space)
    amount_to_surrogate_score = 1000
    candidates_to_score = random_candidates[:amount_to_surrogate_score] + \
                          [search_space.get_random_candidate() for _ in range(amount_to_surrogate_score)]

    # choose the criteria
    criteria = 'popularity'
    inverted = False
    selected_features = None
    if criteria == 'fitness':
        selected_features = fit_features + unfit_features
    elif criteria == 'popularity':
        selected_features = popular_features
    elif criteria == 'all':
        selected_features = fd.get_explainable_features(criteria='all')
        selected_features = [random.choice(selected_features) for _ in range(search_space.total_cardinality * 2)]
    else:
        print("You mistyped.")
        return

    (selected_features, _) = utils.unzip(selected_features)

    print(f"Your selected criteria is {criteria}")

    # then we prepare the feature_presence_matrix, like a rube goldberg machine ...
    featureH_pool = selected_features
    candidate_matrix = hot_encoder.to_hot_encoded_matrix(random_candidates)
    feature_presence_matrix = variate_models.get_feature_presence_matrix(candidate_matrix, featureH_pool)

    # we obtain the appropriate model
    model_matrix = variate_models.get_bivariate_fitness_observations(feature_presence_matrix, np.array(scores))

    def get_surrogate_score(candidateC):
        return variate_models.get_surrogate_score_from_bivariate_model(candidateC, model_matrix, selected_features)

    print("Surrogate evaluation section ####################################")
    amount_of_features = len(selected_features)
    print(f"Some info: we are considering {amount_of_features} features")
    print(f"\t This means that the feature detection matrix is {search_space.total_cardinality} x {amount_of_features}")
    print(f"\t And the surrogate score matrix is {amount_of_features} x {amount_of_features}")

    """print("Some test data: ")
    for candidateC in candidates_to_score:
        surrogate_score = get_surrogate_score(candidateC)
        real_score = problem.score_of_candidate(candidateC)
        #problem.pretty_print_feature(candidateC)
        print(f"{surrogate_score}\t{real_score}")
        #print()
        
    """


def test_surrogate_scorer(problem):
    print(f"The problem is {problem}")

    search_space = problem.get_search_space()
    (training_candidates, training_scores) = get_problem_training_data(problem, 1000)

    # parameters
    importance_of_explainability = 0.5
    complexity_damping = 1
    merging_power = 5

    feature_discoverer = Version_B.FeatureDiscoverer.\
                        FeatureDiscoverer(search_space=search_space, candidateC_population=training_candidates,
                                          fitness_scores=training_scores, merging_power=merging_power,
                                          complexity_function=problem.get_complexity_of_feature,
                                          complexity_damping=complexity_damping,
                                          importance_of_explainability=importance_of_explainability)

    print("Exploring features...")
    feature_discoverer.generate_explainable_features()
    print("Obtaining the fit and unfit features")
    (fit_features, unfit_features) = feature_discoverer.get_explainable_features(criteria='fitness')
    fit_features = utils.unzip(fit_features)[0]
    unfit_features = utils.unzip(unfit_features)[0]

    print("The fit features are:")
    pretty_print_features(problem, fit_features)

    print("The unfit features are:")
    pretty_print_features(problem, unfit_features)

    print("Instantiating the surrogate scorer")
    scorer = SurrogateScorer(model_power=2,
                             search_space=search_space,
                             featuresH=fit_features+unfit_features)
    print("And now we train the model")
    scorer.train(training_candidates, training_scores)
    scorer.make_picky()

    print(f"The model is now {scorer}")


    def sanity_check():
        test_candidate = search_space.get_random_candidate()
        test_score = problem.score_of_candidate(test_candidate)
        surrogate_score = scorer.get_surrogate_score_of_fitness(test_candidate)
        surrogate_mistrustful_score = scorer.get_surrogate_score_of_fitness(test_candidate, based_on_trust=True)

        print(f"For a randomly generated candidate with actual score {test_score}, the surrogate score is {surrogate_score}")

    def print_data_for_analysis():
        (test_candidates, test_scores) = get_problem_training_data(problem, 1000)

        for (test_candidate, test_score) in zip(test_candidates, test_scores):
            surrogate_score = scorer.old_get_surrogate_score_of_fitness(test_candidate)
            surrogate_mistrustful_score = scorer.get_surrogate_score_of_fitness(test_candidate, based_on_trust=True)
            print(f"{test_score}\t{surrogate_score}\t{surrogate_mistrustful_score}")

    sanity_check()
    print_data_for_analysis()



if __name__ == '__main__':
    test_surrogate_scorer(trap5)


# TODO
# fix model power > 2 not producing the correct diagonal exclusion matrix
# rewrite the picky / not picky implementation, perhaps just subtract the diagonal!
# investigate why the scores are so bad
        # is it because not enough features are used?
        # test by changing the arrangement of the cells in binval
# implement variance modelling
# implement variance weighted modelling
# write proof of KNN equivalence