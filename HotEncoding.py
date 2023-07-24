import utils
import numpy as np
import SearchSpace


def candidate_contains_feature(candidate, feature_hot):
    # it's equivalent to NOT(ANY(NOT(C)&F))
    # which can be calculated as 0==SUM((1-C)&F)
    return 0 == np.sum((1 - candidate) & feature_hot)


def get_search_space_flat_clash_matrix(search_space: SearchSpace.SearchSpace):
    unflattened = np.zeros((search_space.total_cardinality, search_space.total_cardinality))

    def set_clash_matrix_for_variable(var_index):
        var_start = search_space.precomputed_offsets[var_index]
        var_end = search_space.precomputed_offsets[var_index + 1]
        unflattened[var_start:var_end, var_start:var_end] = 1 - np.identity(search_space.cardinalities[var_index])

    for var_index in range(search_space.dimensions):
        set_clash_matrix_for_variable(var_index)

    return unflattened.ravel()


def fast_features_are_invalid(featureH_matrix, flat_clash_matrix):
    flat_outer_for_each_feature = utils.row_wise_self_outer_product(featureH_matrix)
    return flat_outer_for_each_feature @ flat_clash_matrix


def feature_is_valid_in_search_space(feature_hot, search_space: SearchSpace.SearchSpace):
    """This function returns whether the hot encoded feature is valid"""
    """It's written horribly because I was trying to make it efficient."""
    start_index = 0
    end_index = search_space.precomputed_offsets[0]
    for i in range(search_space.dimensions):
        start_index = end_index
        end_index = search_space.precomputed_offsets[i + 1]
        if np.sum(feature_hot[start_index:end_index]) > 1.0:
            return False
    return True

    def sum_between(start_index, end_index):
        return np.sum(feature_hot[start_index:end_index])

    return all([sum_between(begin, end) < 2
                for (begin, end) in utils.adjacent_pairs(search_space.precomputed_offsets)])


class HotEncoder:
    search_space: SearchSpace

    def __init__(self, search_space):
        self.search_space = search_space

    @property
    def empty_feature(self):  # TO TEST
        return np.zeros(sum(self.search_space.cardinalities))

    def to_hot_encoding(self, candidate):
        """works for both candidates and features"""
        return np.concatenate(
            [utils.one_hot_encoding(candidate.values[var], cardinality)
             for (var, cardinality) in enumerate(self.search_space.cardinalities)])

    def deconcat_hot_encoding(self, hot_encoded):
        return [hot_encoded[begin:end]
                for (begin, end) in utils.adjacent_pairs(self.search_space.precomputed_offsets)]

    def decode_tuple(self, hot_encoded):
        deconcatted = self.deconcat_hot_encoding(hot_encoded)
        return tuple(utils.from_hot_encoding(hot) for hot in deconcatted)

    def candidate_from_hot_encoding(self, hot_encoded):
        return SearchSpace.Candidate(self.decode_tuple(hot_encoded))

    def feature_from_hot_encoding(self, hot_encoded):
        return SearchSpace.Feature(self.decode_tuple(hot_encoded))

    def to_hot_encoded_matrix(self, populationC):
        return np.array([self.to_hot_encoding(c) for c in populationC])

    def get_hot_encoded_trivial_features(self):
        return [self.to_hot_encoding(feature) for feature in self.search_space.get_all_trivial_features()]

    def is_valid_hot_encoded_feature(self, hot_feature):
        deconcatted = self.deconcat_hot_encoding(hot_feature)

        def is_valid_one_hot_encoding(hot_enc):
            return np.sum(hot_enc) < 2

        return all([is_valid_one_hot_encoding(enc) for enc in deconcatted])


def merge_features(feature_hot_a, feature_hot_b):
    return np.minimum(feature_hot_a + feature_hot_b, 1)


def merge_many_features(list_of_hot_features):
    return np.minimum(np.sum(list_of_hot_features, axis=0), 1)


def features_are_equal(raw_a, raw_b):
    return np.array_equal(raw_a, raw_b)
