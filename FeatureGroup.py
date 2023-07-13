import HotEncoding
import SearchSpace
import numpy as np
import utils


class FeatureGroup:
    """holds a list of features, and detects which are present in candidates"""
    search_space: SearchSpace

    def __init__(self, search_space, hot_features):
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)

        # calculate precomputed feature data
        self.amount_of_features = len(hot_features)
        self.hot_features = hot_features
        self.precomputed_feature_matrix = np.transpose(np.array(self.hot_features))

    def get_feature_vectors_for_candidates(self, candidate_matrix):
        """1 = feature is present"""
        positive_when_absent = (1 - candidate_matrix) @ self.precomputed_feature_matrix
        return 1 - np.minimum(positive_when_absent, 1)

    def get_feature_vector_for_candidate(self, candidate_vector):
        candidate_as_matrix = utils.array_to_row_vector(candidate_vector)
        feature_vector_matrix = self.get_feature_vectors_for_candidates(candidate_as_matrix)
        return feature_vector_matrix[0]

    def get_raw_hot_encoding_from_feature_vector(self, feature_vector):
        sum_of_columns = utils.weighted_sum_of_columns(feature_vector, self.precomputed_feature_matrix)
        return utils.matrix_to_array(np.transpose(np.minimum(sum_of_columns, 1)))
