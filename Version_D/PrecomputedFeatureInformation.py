import numpy as np
import Version_D.PrecomputedPopulationInformation as PPI
from Version_D.Feature import Feature
from typing import Iterable, Optional
from Version_D.HotEncoding import get_hot_encoded_feature
import utils


class PrecomputedFeatureInformation:
    """this data structures stores matrices that are used around the other classes"""
    precomputed_population_information: PPI
    features: list[Feature]
    feature_matrix: np.ndarray
    feature_presence_error_matrix: np.ndarray
    feature_presence_matrix: np.ndarray
    amount_of_features: int

    precomputed_population_mean: Optional[float]
    precomputed_count_for_each_feature: Optional[np.ndarray]
    precomputed_sd_for_each_feature: Optional[np.ndarray]
    precomputed_mean_fitness_for_each_feature: Optional[np.ndarray]
    precomputed_marginal_probabilities: Optional[np.ndarray]

    @property
    def search_space(self):
        return self.precomputed_population_information.search_space

    @property
    def candidate_matrix(self):
        return self.precomputed_population_information.candidate_matrix

    @property
    def sample_size(self):
        return self.precomputed_population_information.sample_size

    @property
    def fitness_array(self) -> np.ndarray:
        return self.precomputed_population_information.fitness_array

    def compute_feature_matrix(self, features: Iterable[Feature]) -> np.ndarray:
        hot_encoded_features = [get_hot_encoded_feature(feature, self.search_space) for feature in features]
        return np.array(hot_encoded_features)

    def compute_feature_presence_error_matrix(self):
        errors = (1 - self.candidate_matrix) @ self.feature_matrix
        # the (row = i, col = j) element of errors is the amount of errors present when checking how well
        # feature # j fits in candidate # i
        # In other words, each row is a candidate as the features it contains.
        return errors

    def compute_feature_presence_matrix(self) -> np.ndarray:
        return np.array(self.feature_presence_error_matrix < 1, dtype=float)

    def compute_population_mean(self) -> float:
        return np.mean(self.fitness_array)

    def compute_count_for_each_feature(self) -> np.ndarray:
        return np.sum(self.feature_presence_matrix, axis=0)

    def compute_mean_fitness_for_each_feature(self) -> np.ndarray:
        sum_of_fitnesses = utils.weighted_sum_of_rows(self.feature_presence_matrix,
                                                      self.fitness_array)

        return utils.divide_arrays_safely(sum_of_fitnesses, self.count_for_each_feature)

    def compute_sd_for_each_feature(self) -> np.ndarray:
        numerators = utils.weighted_sum_of_rows(self.feature_presence_matrix,
                                                np.square(self.fitness_array - self.population_mean))

        return np.sqrt(utils.divide_arrays_safely(numerators, (self.count_for_each_feature - 1)))


    def compute_marginal_probabilities(self) -> np.ndarray:
        sum_in_hot_encoding_order: np.ndarray[float] = np.sum(self.candidate_matrix, axis=0)
        return sum_in_hot_encoding_order / self.sample_size

    @property
    def count_for_each_feature(self) -> np.ndarray:
        if not self.precomputed_count_for_each_feature:
            self.precomputed_count_for_each_feature = self.compute_count_for_each_feature()
        return self.precomputed_count_for_each_feature

    @property
    def mean_fitness_for_each_feature(self) -> np.ndarray:
        if not self.precomputed_mean_fitness_for_each_feature:
            self.precomputed_mean_fitness_for_each_feature = self.compute_mean_fitness_for_each_feature()
        return self.precomputed_mean_fitness_for_each_feature

    @property
    def sd_for_each_feature(self) -> np.ndarray:
        if not self.precomputed_sd_for_each_feature:
            self.precomputed_sd_for_each_feature = self.compute_sd_for_each_feature()
        return self.precomputed_sd_for_each_feature

    @property
    def population_mean(self) -> float:
        if not self.precomputed_population_mean:
            self.precomputed_population_mean = self.compute_population_mean()
        return self.precomputed_population_mean


    @property
    def marginal_probabilities(self) -> np.ndarray:
        if not self.precomputed_marginal_probabilities:
            self.precomputed_marginal_probabilities = self.compute_marginal_probabilities()
        return self.precomputed_marginal_probabilities

    def __init__(self, population_precomputed: PPI,
                 features: Iterable[Feature]):
        self.precomputed_population_information = population_precomputed
        self.features = list(features)
        self.feature_matrix = self.compute_feature_matrix(features)
        self.feature_presence_error_matrix = self.compute_feature_presence_error_matrix()
        self.feature_presence_matrix = self.compute_feature_presence_matrix()

        self.precomputed_count_for_each_feature = None
        self.precomputed_population_mean = None
        self.precomputed_mean_fitness_for_each_feature = None
        self.precomputed_sd_for_each_feature = None
        self.precomputed_marginal_probabilities = None

    """
    def get_t_scores(self) -> np.ndarray:
        means = self.get_average_fitness_vector()
        overall_average = self.get_overall_average_fitness()

        #  this is the sum of (mean (of each feature) minus the overall mean)**2
        numerators = utils.weighted_sum_of_rows(self.feature_presence_matrix,
                                                np.square(self.fitness_array - overall_average))

        standard_deviations = np.sqrt(utils.divide_arrays_safely(numerators, (self.count_for_each_feature - 1)))

        sd_over_root_n = utils.divide_arrays_safely(standard_deviations, np.sqrt(self.count_for_each_feature))
        t_scores = utils.divide_arrays_safely(means - overall_average, sd_over_root_n)

        return t_scores

    def get_distance_in_fitness_with_one_change(self) -> np.ndarray:
        normal_means = self.get_average_fitness_vector()
        off_by_one_means = VariateModels.get_means_from_fpm(self.get_off_by_one_feature_presence_matrix(),
                                                            self.fitness_array)

        return (normal_means - off_by_one_means) / (1 + np.abs(normal_means) + np.abs(off_by_one_means))
"""
