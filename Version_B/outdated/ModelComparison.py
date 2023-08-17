from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from BenchmarkProblems import CombinatorialProblem
from Version_B import VariateModels
from Version_B.SurrogateScorer import SurrogateScorer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from Version_B.Sampler import Sampler
import numpy as np
import csv
import datetime


""" This file is just a code template to test many scikitlearn models"""



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


def get_problem_training_data(problem, param):
    pass


def get_explainable_features(problem):
    pass


def test_surrogate_model(problem: CombinatorialProblem.CombinatorialProblem):
    search_space = problem.search_space
    training_data = get_problem_training_data(problem, 2000)
    (fit_features, unfit_features, pop_features, unpop_features) = get_explainable_features(problem)

    sampler = Sampler(search_space, fit_features, unfit_features, unpop_features,
                      importance_of_novelty=0)

    sampler.train(*training_data)

    print("We can generate some new candidates:")
    for _ in range(12):
        new_candidate = search_space.get_random_candidate()
        score = problem.score_of_candidate(new_candidate)
        problem.pretty_print_candidate(new_candidate)
        print(f"(has actual score of {score})\n")

    detectors = [VariateModels.FeatureDetector(search_space, feature_list)
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
    own_regressor = SurrogateScorer(model_power=2, search_space=search_space,
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