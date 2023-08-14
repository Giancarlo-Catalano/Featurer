import numpy as np

import BenchmarkProblems.CombinatorialProblem
import SearchSpace
import HotEncoding
import utils
import VariateModels


class SampleAndPrecomputedData:
    candidate_matrix: np.ndarray
    feature_presence_matrix: np.ndarray
    fitness_array: np.ndarray
    featuresH: list[np.ndarray]

    def __init__(self, candidate_matrix, fitness_array):
        self.candidate_matrix = candidate_matrix
        self.fitness_array = fitness_array
        self.featuresH = None
        self.feature_presence_matrix = None

    def set_features(self, featuresH: list[np.ndarray]):
        self.featuresH = featuresH
        feature_matrix = np.transpose(np.array(self.featuresH))
        positive_when_absent = (1 - self.candidate_matrix) @ feature_matrix
        self.feature_presence_matrix = 1 - np.minimum(positive_when_absent, 1)

class FeatureFinder:
    problem: BenchmarkProblems.CombinatorialProblem.CombinatorialProblem
    importance_of_explainability: float
    depth: int
    search_space: SearchSpace.SearchSpace
    hot_encoder: HotEncoding.HotEncoder
    variate_model_generator: VariateModels.VariateModels

    sample_and_precomputed_data: SampleAndPrecomputedData
    considered_features: list[SearchSpace.Feature]

    def __init__(self, problem, depth, importance_of_explainability):
        self.problem = problem
        self.depth = depth
        self.importance_of_explainability = importance_of_explainability

        self.search_space = self.problem.search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.trivial_features = self.search_space.get_all_trivial_features()
        self.variate_model_generator = VariateModels.VariateModels(self.search_space)

        self.sample_and_precomputed_data = None
        self.considered_features = []

    def set_population(self, population: list[SearchSpace.Candidate]):
        candidate_matrix = self.hot_encoder.to_hot_encoded_matrix(population)
        fitness_array = np.ndarray([self.get_fitness_of_candidate(candidate) for candidate in population])
        self.sample_and_precomputed_data = SampleAndPrecomputedData(candidate_matrix, fitness_array)

    def get_fitness_of_candidate(self, candidate):
        return self.problem.score_of_candidate(candidate)

    def get_complexity_of_feature(self, feature):
        return self.problem.get_complexity_of_feature(feature)

    def get_explainability_relevance_of_features(self) -> np.ndarray:
        """
        Returns an array of values from 0 to 1 which are associated with explainability, the opposite of complexity
        :return: an array, where the scores are in the same order as the input list.
        """
        complexities = np.array([self.get_complexity_of_feature(feature) for feature in self.considered_features])
        normalised_complexities = utils.remap_array_in_zero_one(complexities)
        return 1.0 - normalised_complexities

    def cache_features_in_population_sample(self):
        self.sample_and_precomputed_data.set_features([self.hot_encoder.feature_to_hot_encoding(feature)
                                                       for feature in self.considered_features])
    def get_fitness_relevance_of_features(self):
        """
        Returns an array of values from 0 to 1 which correspond to how strongly each feature affects the fitness,
        which might be positively or negatively. This only measures how strong it is!
        :return: an np.ndarray containing all of the values, where 0 means irrelevant, 1 means very
        interesting
        """
        return (self.variate_model_generator
                .get_average_fitness_of_features_from_matrix(self.sample_and_precomputed_data.feature_presence_matrix,
                                                             self.sample_and_precomputed_data.fitness_array))

    def get_overall_scores_of_features(self):
        explainabilities = self.get_explainability_relevance_of_features()
        fitness_relevances = self.get_fitness_relevance_of_features()
        return utils.weighted_sum(explainabilities, self.importance_of_explainability,
                                  fitness_relevances, 1.0 - self.importance_of_explainability)

    def get_the_best_features(self, how_many_to_keep: int):
        scores = self.get_overall_scores_of_features()

        sorted_by_with_score = sorted(zip(self.considered_features, scores), key=utils.second, reverse=True)
        return sorted_by_with_score[:how_many_to_keep]





