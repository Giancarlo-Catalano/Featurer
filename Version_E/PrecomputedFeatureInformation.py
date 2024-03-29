import numpy as np
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Feature import Feature
from typing import Iterable, Optional
from Version_E.HotEncoding import get_hot_encoded_feature
import utils


class PrecomputedFeatureInformation:
    """this data structures stores matrices that are used around the other classes"""
    precomputed_population_information: PrecomputedPopulationInformation
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
    precomputed_normalised_fitnesses: Optional[np.ndarray]

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
        return np.array(hot_encoded_features).T

    def compute_feature_presence_error_matrix(self):
        inverted_candidate_matrix = 1- self.candidate_matrix  # (self.candidate_matrix == 0).astype(int) #
        feature_matrix = self.feature_matrix  # self.feature_matrix.astype(int) #
        errors = inverted_candidate_matrix @ feature_matrix
        # the (row = i, col = j) element of errors is the amount of errors present when checking how well
        # feature # j fits in candidate # i
        # In other words, each row is a candidate as the features it contains.
        return errors # .astype(float)

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
        # NOTE: the rows of a feature presence matrix represent candidates, the columns represent features

        fitness_of_candidate_minus_feature_mean = np.subtract.outer(self.fitness_array,
                                                                    self.mean_fitness_for_each_feature)

        fitness_of_candidate_minus_feature_mean *= self.feature_presence_matrix
        fitness_of_candidate_minus_feature_mean = np.square(fitness_of_candidate_minus_feature_mean)

        numerators = np.sum(fitness_of_candidate_minus_feature_mean, axis=0)

        return np.sqrt(utils.divide_arrays_safely(numerators, self.count_for_each_feature-1))

    def compute_marginal_probabilities(self) -> np.ndarray:
        sum_in_hot_encoding_order: np.ndarray[float] = np.sum(self.candidate_matrix, axis=0)
        return sum_in_hot_encoding_order / self.sample_size


    def compute_normalised_fitnesses(self) -> np.ndarray:
        means = self.mean_fitness_for_each_feature
        min_fitness = np.min(self.fitness_array)
        max_fitness = np.max(self.fitness_array)

        position_within_range = (means - min_fitness) / (max_fitness - min_fitness)

        return position_within_range

    @property
    def amount_of_features(self):
        return len(self.features)

    @property
    def count_for_each_feature(self) -> np.ndarray:
        if self.precomputed_count_for_each_feature is None:
            self.precomputed_count_for_each_feature = self.compute_count_for_each_feature()
        return self.precomputed_count_for_each_feature

    @property
    def mean_fitness_for_each_feature(self) -> np.ndarray:
        if self.precomputed_mean_fitness_for_each_feature is None:
            self.precomputed_mean_fitness_for_each_feature = self.compute_mean_fitness_for_each_feature()
        return self.precomputed_mean_fitness_for_each_feature

    @property
    def sd_for_each_feature(self) -> np.ndarray:
        if self.precomputed_sd_for_each_feature is None:
            self.precomputed_sd_for_each_feature = self.compute_sd_for_each_feature()
        return self.precomputed_sd_for_each_feature

    @property
    def population_mean(self) -> float:
        if self.precomputed_population_mean is None:
            self.precomputed_population_mean = self.compute_population_mean()
        return self.precomputed_population_mean

    @property
    def marginal_probabilities(self) -> np.ndarray:
        if self.precomputed_marginal_probabilities is None:
            self.precomputed_marginal_probabilities = self.compute_marginal_probabilities()
        return self.precomputed_marginal_probabilities

    def __init__(self, population_precomputed: PrecomputedPopulationInformation,
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

    @classmethod
    def get_dummy_pfi(cls, ppi: PrecomputedPopulationInformation):
        dummy_features = [Feature.empty_feature(ppi.search_space)]
        return cls(ppi, dummy_features)

    @classmethod
    def get_from_hot_encoded_features(cls, ppi: PrecomputedPopulationInformation, hot_encoded_features: np.ndarray):
        """I'm quite ashamed of this method, because it's against everything I believe in
           ... but it's convenient...
           Its purpose is to give the ability to get information about features without the requirement
           of them having to be the fully fledged Feature objects"""
        result_pfi = cls.get_dummy_pfi(ppi)
        amount_of_features, amount_of_columns = hot_encoded_features.shape

        # note that result_pfi.features will be invalid

        result_pfi.feature_matrix = hot_encoded_features.T
        result_pfi.feature_presence_error_matrix = result_pfi.compute_feature_presence_error_matrix()
        result_pfi.feature_presence_matrix = result_pfi.compute_feature_presence_matrix()
        #result_pfi.amount_of_features = amount_of_features
        return result_pfi


