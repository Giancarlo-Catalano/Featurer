import HotEncoding
import SearchSpace
import utils
import numpy as np


class FeatureDetector:
    """An object which can be used to quickly detect which features are present in a list of candidates"""
    search_space: SearchSpace.SearchSpace
    hot_encoder: HotEncoding.HotEncoder
    detection_matrix: np.ndarray

    def __init__(self, search_space, featureH_pool):
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.detection_matrix = np.transpose(np.array(featureH_pool))

    @property
    def amount_of_features(self):
        return self.detection_matrix.shape[1]

    def __repr__(self):
        (total_cardinality, amount_of_features) = self.detection_matrix.shape
        return f"FeatureDetector(detection_matrix : {total_cardinality} x {amount_of_features})"

    def get_feature_presence_matrix_from_candidate_matrix(self, candidate_matrix):
        """returns a matrix in which element i, j is 1 if candidate i contains feature j"""
        positive_when_absent = (1 - candidate_matrix) @ self.detection_matrix
        return 1 - np.minimum(positive_when_absent, 1)

    def get_feature_presence_matrix_from_candidates(self, candidatesC):
        """this might get used especially to process training data in batches"""
        candidate_matrix = self.hot_encoder.to_hot_encoded_matrix(candidatesC)
        return self.get_feature_presence_matrix_from_candidate_matrix(candidate_matrix)

    def get_feature_presence_from_candidate(self, candidateC):
        candidate_matrix = self.hot_encoder.to_hot_encoded_matrix([candidateC])
        return self.get_feature_presence_matrix_from_candidate_matrix(candidate_matrix).ravel()


class FeatureValidator:
    """ An object throgh which you can quickly check which candidates / features are valid in the given search space"""
    search_space: SearchSpace.SearchSpace
    hot_encoder: HotEncoding.HotEncoder
    clash_matrix: np.ndarray

    def get_search_space_flat_clash_matrix(self, search_space: SearchSpace.SearchSpace):
        unflattened = np.zeros((search_space.total_cardinality, search_space.total_cardinality))

        def set_clash_matrix_for_variable(var_index):
            var_start = search_space.precomputed_offsets[var_index]
            var_end = search_space.precomputed_offsets[var_index + 1]
            unflattened[var_start:var_end, var_start:var_end] = 1 - np.identity(search_space.cardinalities[var_index])

        for var_index in range(search_space.dimensions):
            set_clash_matrix_for_variable(var_index)

        return unflattened.ravel()

    def __init__(self, search_space):
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.clash_matrix = self.get_search_space_flat_clash_matrix(self.search_space)

    def __repr__(self):
        rows = self.search_space.total_cardinality
        return f"FeatureValidator(detection_matrix : ({rows} x {rows})"

    def get_feature_clashing(self, featuresH) -> np.ndarray:
        """

        :param featuresH: a list of featureH
        :return: a np.array, where 1 indicates that the feature is INvalid, and 0 if it is valid
        """
        featureH_matrix = np.array(featuresH)
        flat_outer_for_each_feature = utils.row_wise_self_outer_product(featureH_matrix)
        return flat_outer_for_each_feature @ self.clash_matrix


class VariateModels:
    # features: a list of featuresH
    # correlation matrices

    def get_feature_presence_matrix(self, candidate_matrix, featureH_pool) -> np.ndarray:
        """returns a matrix in which element i, j is 1 if candidate i contains feature j"""
        feature_matrix = np.transpose(np.array(featureH_pool))
        positive_when_absent = (1 - candidate_matrix) @ feature_matrix
        return 1 - np.minimum(positive_when_absent, 1)

    def clean_criteria_scores_given_expectations(self, criteria_scores: np.ndarray, expectations: np.ndarray):
        chi_squared_and_is_good = [(utils.chi_squared(observed, expected),
                                    observed > expected)
                                   for (observed, expected) in zip(criteria_scores, expectations)]

        (goodness, badness) = utils.unzip([(x_2, 0.0) if is_good else (0.0, x_2)
                                           for (x_2, is_good)
                                           in chi_squared_and_is_good])

        result = (utils.remap_array_in_zero_one(goodness), utils.remap_array_in_zero_one(badness))
        return result

    def get_fitness_unfitness_scores(self, feature_presence_matrix, fitness_list) -> (np.ndarray, np.ndarray):
        """returns the univariate model according to the fitness"""
        count_for_each_feature = np.sum(feature_presence_matrix, axis=0)
        sum_of_fitness_for_each_feature = np.sum(feature_presence_matrix * utils.to_column_vector(fitness_list),
                                                 axis=0)

        average_fitnesses = np.array([total / float(count) if count > 0 else 0.0 for total, count in
                                      zip(sum_of_fitness_for_each_feature, count_for_each_feature)])

        average_fitness_overall = np.mean(fitness_list)
        expectation_list = [average_fitness_overall for _ in average_fitnesses]
        return self.clean_criteria_scores_given_expectations(average_fitnesses, expectation_list)

    def get_popularity_unpopularity_scores(self, feature_presence_matrix, expected_amounts) -> (list, list):
        """returns the univariate model according to the popularity"""
        observed_amounts = np.sum(feature_presence_matrix, axis=0)
        return self.clean_criteria_scores_given_expectations(observed_amounts, expected_amounts)

    def get_fitness_unstability_scores(self, feature_presence_matrix, fitness_list):
        def standard_deviation(observed_array, presence_array):
            amount = np.sum(presence_array)
            if amount < 2:
                return 0.0
            mean = np.sum(observed_array)/amount
            # i have to multiply by the presence array, to only remove the appropriate cases.
            numerator = np.sum(np.square(observed_array - presence_array * mean))
            standard_deviation = np.sqrt(numerator / (amount - 1))
            return standard_deviation / mean


        observations = feature_presence_matrix * utils.to_column_vector(
            fitness_list)  # TODO this is calculated elsewhere, perhaps it can be cached?

        result_list = []

        for observed_fitnesses, presences in zip(observations.T, feature_presence_matrix.T):
            result_list.append(standard_deviation(observed_fitnesses, presences))

        return np.array(result_list)

    def get_bivariate_fitness_observations(self, feature_presence_matrix, fitness_list: np.ndarray):
        row_wise_outer_product = utils.row_wise_self_outer_product(feature_presence_matrix)
        weighted_sum_by_fitness = np.sum(row_wise_outer_product * utils.to_column_vector(fitness_list),
                                         axis=0)  # gravity sum

        count_for_each_pair_of_features = np.sum(row_wise_outer_product, axis=0)
        average_fitnesses = np.array([total / float(count) if count > 0 else 0.0 for total, count in
                                      zip(weighted_sum_by_fitness, count_for_each_pair_of_features)])

        # average fitnesses needs to be re ranged and then reshaped
        cooccurrence_matrix = utils.remap_array_in_zero_one_ignore_zeros(
            average_fitnesses)  # TODO this is also looking at the zeros, whereas it should really ignore them!
        amount_of_features = feature_presence_matrix.shape[1]
        return cooccurrence_matrix.reshape((amount_of_features, amount_of_features))

    def get_bivariate_popularity_observations(self, feature_presence_matrix):
        row_wise_outer_product = utils.row_wise_self_outer_product(feature_presence_matrix)
        count_for_each_pair_of_features = np.sum(row_wise_outer_product, axis=0)

        # average fitnesses needs to be re ranged and then reshaped
        cooccurrence_matrix = utils.remap_array_in_zero_one(count_for_each_pair_of_features)
        amount_of_features = feature_presence_matrix.shape[1]
        return cooccurrence_matrix.reshape((amount_of_features, amount_of_features))

    def get_average_fitness_of_features(self, feature_presence_matrix, fitness_list):
        count_for_each_feature = np.sum(feature_presence_matrix, axis=0)
        sum_of_fitness_for_each_feature = np.sum(feature_presence_matrix * utils.to_column_vector(fitness_list),
                                                 axis=0)

        average_fitnesses = np.array([total / float(count) if count > 0 else 0.0 for total, count in
                                      zip(sum_of_fitness_for_each_feature, count_for_each_feature)])

        return average_fitnesses

    def get_univariate_popularity_model(self, feature_presence_matrix):
        count_for_each_feature = np.sum(feature_presence_matrix, axis=0)
        return count_for_each_feature

    def get_expected_bivariate_model_from_marginals(self, marginals):
        return np.outer(marginals, marginals)

    def get_surrogate_score_from_bivariate_model(self, candidateC, bivariate_model_matrix, model_features):
        candidateH = self.hot_encoder.to_hot_encoding(candidateC)
        candidates_features = self.get_feature_presence_matrix(candidateH, model_features)
        row = candidates_features
        column = utils.as_column_matrix(candidates_features)
        sum_of_cells = ((row @ bivariate_model_matrix) @ column)[0]
        amount_of_cells = np.sum(candidates_features) ** 2
        if amount_of_cells == 0:
            return 0.0
        return sum_of_cells / amount_of_cells

    def __init__(self, search_space: SearchSpace.SearchSpace):
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(search_space)
