import datetime

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import BenchmarkProblems.CombinatorialProblem
import HotEncoding
import SearchSpace
import numpy as np
import utils
from BenchmarkProblems import CombinatorialProblem, CheckerBoard, OneMax, BinVal, TrapK, BT, GraphColouring

import Version_B.outdated.FeatureDiscoverer
from Version_B.Sampler import ESTEEM_Sampler
from Version_B.SurrogateScorer import SurrogateScorer
import Version_B.FeatureExplorer
from Version_B.VariateModels import VariateModels
import Version_B.FeatureFinder
import csv

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

trap5 = TrapK.TrapK(5, 3)
checkerboard = CheckerBoard.CheckerBoardProblem(5, 5)
onemax = OneMax.OneMaxProblem(12)
binval = BinVal.BinValProblem(12, 2)
BT = BT.BTProblem(25, 3)
graph_colouring = GraphColouring.GraphColouringProblem(3, 6, 0.5)

depth = 5
importance_of_explainability = 0.5


def get_problem_training_data(problem: CombinatorialProblem.CombinatorialProblem, sample_size):
    search_space = problem.search_space
    random_candidates = [search_space.get_random_candidate() for _ in range(sample_size)]

    scores = [problem.score_of_candidate(c) for c in random_candidates]
    return random_candidates, scores


def pretty_print_features(problem: CombinatorialProblem.CombinatorialProblem, input_list_of_features, with_scores=False,
                          combinatorial=False):
    """prints the passed features, following the structure specified by the problem"""
    hot_encoder = HotEncoding.HotEncoder(problem.search_space)

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
        print("\n")


def without_scores(features_with_scores):
    return utils.unzip(features_with_scores)[0]


def hot_encoded_features(problem, featuresC):
    hot_encoder = HotEncoding.HotEncoder(problem.search_space)
    return [hot_encoder.feature_to_hot_encoding(featureC) for featureC in featuresC]


def output_onto_csv_file(filename, headers, generator_function, how_many_samples):
    print(f"Writing the some output data onto {filename}")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows([generator_function() for _ in range(how_many_samples)])


def get_appropriate_filename(problem: CombinatorialProblem.CombinatorialProblem, tags=[]):
    directory = "outputs\\"
    timestamp = datetime.datetime.now().strftime("%H'%M %d-%m-%Y")
    tags_together = "[" + "|".join(tags) + "]"
    extension = ".csv"
    return f"{directory}{problem} -- {timestamp}{tags_together}{extension}"


def get_explainable_features(problem, training_data):
    search_space = problem.search_space
    print("Testing the explainable feature explorer")
    print(f"The problem is {problem}")

    print("Constructing the explorer")
    explorer = Version_B.FeatureExplorer.FeatureExplorer(search_space, depth, problem.get_complexity_of_feature,
                                                         importance_of_explainability=importance_of_explainability)

    training_samples, training_scores = training_data

    print("Then we obtain the meaningful features")
    (fit_features_and_scores), (unfit_features_and_scores), (popular_features_and_scores), (
        unpopular_features_and_scores) = \
        explorer.get_important_explainable_features(training_samples, training_scores)

    print("The fit features are")
    pretty_print_features(problem, fit_features_and_scores, combinatorial=True, with_scores=True)

    print("\n\nThe unfit features are")
    pretty_print_features(problem, unfit_features_and_scores, combinatorial=True, with_scores=True)

    print("\n\nThe popular features are")
    pretty_print_features(problem, popular_features_and_scores, combinatorial=True, with_scores=True)

    print("\n\nThe unpopular features are")
    pretty_print_features(problem, unpopular_features_and_scores, combinatorial=True, with_scores=True)

    features_per_category = search_space.dimensions

    (fit_features,
     unfit_features,
     pop_features,
     unpop_features) = (hot_encoded_features(problem, without_scores(features_and_scores))[:features_per_category]
                        for features_and_scores in (fit_features_and_scores,
                                                    unfit_features_and_scores,
                                                    popular_features_and_scores,
                                                    unpopular_features_and_scores))
    return (fit_features, unfit_features, pop_features, unpop_features)


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
    search_space = problem.search_space
    print("Testing the explainable feature explorer")
    print(f"The problem is {problem}")

    print("Constructing the explorer")
    explorer = Version_B.FeatureExplorer.FeatureExplorer(search_space, depth, problem.get_complexity_of_feature,
                                                         importance_of_explainability=importance_of_explainability)

    training_samples, training_scores = get_problem_training_data(problem, 200)

    print("Then we obtain the meaningful features")
    (fit_features_and_scores), (unfit_features_and_scores), (popular_features_and_scores), (
        unpopular_features_and_scores) = \
        explorer.get_important_explainable_features(training_samples, training_scores)

    print("The fit features are")
    pretty_print_features(problem, fit_features_and_scores, combinatorial=True, with_scores=True)

    print("\n\nThe unfit features are")
    pretty_print_features(problem, unfit_features_and_scores, combinatorial=True, with_scores=True)

    print("\n\nThe popular features are")
    pretty_print_features(problem, popular_features_and_scores, combinatorial=True, with_scores=True)

    print("\n\nThe unpopular features are")
    pretty_print_features(problem, unpopular_features_and_scores, combinatorial=True, with_scores=True)

    features_per_category = search_space.dimensions

    (fit_features,
     unfit_features,
     pop_features,
     unpop_features) = (hot_encoded_features(problem, without_scores(features_and_scores))[:features_per_category]
                        for features_and_scores in (fit_features_and_scores,
                                                    unfit_features_and_scores,
                                                    popular_features_and_scores,
                                                    unpopular_features_and_scores))

    print("Then we sample some new candidates")
    sampler = ESTEEM_Sampler(search_space, fit_features=fit_features,
                             unfit_features=unfit_features, unpopular_features=unpop_features,
                             importance_of_novelty=0.1)

    sampler.train(training_samples, training_scores)

    for _ in range(10):
        new_candidate = sampler.sample()
        actual_score = problem.score_of_candidate(new_candidate)

        problem.pretty_print_candidate(new_candidate)
        print(f"Has score {actual_score}\n")

    print("We can then analyse the unstabilities")
    features_to_be_considered = fit_features + unfit_features + pop_features
    variate_model_generator = Version_B.VariateModels.VariateModels(search_space)

    hot_encoder = HotEncoding.HotEncoder(search_space)
    candidate_matrix = hot_encoder.to_hot_encoded_matrix(training_samples)
    featuresH = features_to_be_considered
    feature_presence_matrix = variate_model_generator.get_feature_presence_matrix(candidate_matrix, featuresH)

    fitness_array = np.array(training_scores)
    unstabilities = variate_model_generator.get_fitness_unstability_scores(feature_presence_matrix, fitness_array)
    counterstabilities = variate_model_generator.get_fitness_unstability_scores(1.0 - feature_presence_matrix,
                                                                                fitness_array)

    features_with_unstabilities_and_counterstabilities = list(
        zip(features_to_be_considered, unstabilities, counterstabilities))

    features_with_unstabilities_and_counterstabilities.sort(key=lambda x: min(x[1], x[2]))

    for feature, unstability, counterstability in features_with_unstabilities_and_counterstabilities:
        problem.pretty_print_feature(hot_encoder.feature_from_hot_encoding(feature))
        print(f"(Has unstability = {unstability:.2f}, counterstability = {counterstability:.2f}")

    print("We will be training the surrogate scorer using the most stable features")

    def get_scorer_with_features(featuresH):
        return Version_B.SurrogateScorer.SurrogateScorer(2,
                                                         search_space=search_space,
                                                         featuresH=featuresH,
                                                         with_inverse=False)

    def train_scorer(scorer):
        scorer.train(training_samples, training_scores)
        scorer.make_picky()

    def get_surrogate_score(candidate, model):
        return model.get_surrogate_score_of_fitness(candidateC=candidate, based_on_trust=False)

    fit_scorer = get_scorer_with_features(fit_features)
    unfit_scorer = get_scorer_with_features(unfit_features)
    pop_scorer = get_scorer_with_features(pop_features)

    scorers = [fit_scorer, unfit_scorer, pop_scorer]

    print("Training the models...")
    for scorer in scorers:
        train_scorer(scorer)

    print("Generating surrogate scores and comparing them to actual scores")


def test_surrogate_model(problem: CombinatorialProblem.CombinatorialProblem):
    search_space = problem.search_space
    training_data = get_problem_training_data(problem, 2000)
    explainable_features = get_explainable_features(problem, training_data)
    (fit_features, unfit_features, pop_features, unpop_features) = explainable_features

    sampler = Version_B.Sampler.ESTEEM_Sampler(search_space, fit_features, unfit_features, unpop_features,
                                               importance_of_novelty=0)

    sampler.train(*training_data)

    print("We can generate some new candidates:")
    for _ in range(12):
        new_candidate = search_space.get_random_candidate()
        score = problem.score_of_candidate(new_candidate)
        problem.pretty_print_candidate(new_candidate)
        print(f"(has actual score of {score})\n")

    detectors = [Version_B.VariateModels.FeatureDetector(search_space, feature_list)
                 for feature_list in [fit_features, unfit_features, pop_features]]

    def candidate_to_model_input(candidateC):
        [fit_feature_vector,
         unfit_feature_vector,
         popular_feature_vector] = [detector.get_feature_presence_from_candidateC(candidateC)
                                    for detector in detectors]

        return np.concatenate((fit_feature_vector, unfit_feature_vector, popular_feature_vector))

    (original_candidates, scores) = training_data
    datapoints_for_model = [candidate_to_model_input(candidateC) for candidateC in original_candidates]

    X, y = np.array(datapoints_for_model), scores

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # own
    own_regressor = Version_B.SurrogateScorer.SurrogateScorer(model_power=2, search_space=search_space,
                                                              featuresH=fit_features + unfit_features + pop_features)
    own_regressor.train(original_candidates, scores)
    own_regressor.make_picky()

    # decision tree
    decision_tree_regressor = DecisionTreeRegressor(random_state=42)
    decision_tree_regressor.fit(X_train, y_train)

    # Lasso
    lasso_regressor = Lasso(alpha=0.01)
    lasso_regressor.fit(X_train, y_train)

    # Gradient Boosting
    gradient_boosting_regressor = GradientBoostingRegressor(random_state=42)
    gradient_boosting_regressor.fit(X_train, y_train)

    # Random Forest
    random_forest_regressor = RandomForestRegressor(random_state=42)
    random_forest_regressor.fit(X_train, y_train)

    # MLP regressor
    mlp_regressor = MLPRegressor(random_state=42)
    mlp_regressor.fit(X_train, y_train)

    regressors = [decision_tree_regressor, lasso_regressor, gradient_boosting_regressor, random_forest_regressor,
                  mlp_regressor]
    headers = ["actual", "own", "Decision Tree", "Lasso Prediction", "Gradient Boosting", "random_forest_regressor",
               "mlp_regressor"]

    def get_test_datapoint():
        candidate = search_space.get_random_candidate()
        model_input = candidate_to_model_input(candidate)
        model_input = model_input.reshape(1, -1)
        actual_score = problem.score_of_candidate(candidate)
        own_prediction = own_regressor.get_surrogate_score_of_fitness(candidate)
        other_predictions = [regressor.predict(model_input)[0] for regressor in regressors]

        return [actual_score, own_prediction] + other_predictions

    output_onto_csv_file(get_appropriate_filename(problem, tags=["all"]),
                         headers=headers,
                         generator_function=get_test_datapoint,
                         how_many_samples=1000)


def test_finder(problem: BenchmarkProblems.CombinatorialProblem.CombinatorialProblem):
    features, scores = Version_B.FeatureFinder.find_features(problem=problem,
                                                             depth=depth,
                                                             importance_of_explainability=importance_of_explainability,
                                                             heuristic = True,
                                                             sample_size=1000,
                                                             criteria=Version_B.FeatureFinder.ScoringCriterion.HIGH_FITNESS)

    print(f"For the problem {problem}, the found features are:")
    pretty_print_features(problem, features, combinatorial=True)


if __name__ == '__main__':
    problem = trap5
    # print(f"The problem is {problem.long_repr()}")
    test_finder(problem)
