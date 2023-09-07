import utils
import numpy as np
import SearchSpace


def candidate_contains_feature(candidate, feature_hot):
    # it's equivalent to NOT(ANY(NOT(C)&F))
    # which can be calculated as 0==SUM((1-C)&F)
    return 0 == np.sum((1 - candidate) & feature_hot)


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


def hot_encode_candidate(candidate: SearchSpace.Candidate, search_space: SearchSpace)-> np.ndarray:
    return np.concatenate(
        [utils.one_hot_encoding(candidate.values[var], cardinality)
         for (var, cardinality) in enumerate(search_space.cardinalities)])


def hot_encode_feature(feature: SearchSpace.Feature, search_space: SearchSpace) -> np.ndarray:
    result = np.zeros(search_space.total_cardinality, dtype=float)
    for var, val in feature.var_vals:
        result[search_space.precomputed_offsets[var] + val] = 1.0
    return result


def hot_encode_candidate_population(population: list[SearchSpace.Candidate],
                                    search_space: SearchSpace.SearchSpace) -> np.ndarray:
    return np.array([hot_encode_candidate(c, search_space) for c in population])


def hot_encode_feature_list(feature_list: list[SearchSpace.Feature],
                            search_space: SearchSpace.SearchSpace) -> np.ndarray:
    """ returns the feature matrix, transposed already"""
    return np.array([hot_encode_feature(feature, search_space) for feature in feature_list]).T


class HotEncoder:
    search_space: SearchSpace.SearchSpace

    def __init__(self, search_space):
        self.search_space = search_space

    @property
    def empty_feature(self):  # TO TEST
        return np.zeros(sum(self.search_space.cardinalities))

    def candidate_to_hot_encoding(self, candidate):
        return hot_encode_candidate(candidate, self.search_space)

    def feature_to_hot_encoding(self, feature: SearchSpace.Feature):
        return hot_encode_feature(feature, self.search_space)

    def deconcat_hot_encoding(self, hot_encoded):
        return [hot_encoded[begin:end]
                for (begin, end) in utils.adjacent_pairs(self.search_space.precomputed_offsets)]

    def candidate_from_hot_encoding(self, hot_encoded):
        deconcatted = self.deconcat_hot_encoding(hot_encoded)
        value_tuple = tuple(utils.from_hot_encoding(hot) for hot in deconcatted)
        return SearchSpace.Candidate(value_tuple)

    def feature_from_hot_encoding(self, hot_encoded):
        deconcatted = self.deconcat_hot_encoding(hot_encoded)
        decoded = [utils.from_hot_encoding(hot) for hot in deconcatted]
        var_vals = [(index, val) for index, val in enumerate(decoded) if val is not None]
        return SearchSpace.Feature(var_vals)

    def to_hot_encoded_matrix(self, population: list[SearchSpace.Candidate]):
       return hot_encode_candidate_population(population)

    def get_hot_encoded_trivial_features(self):
        return [self.feature_to_hot_encoding(feature) for feature in self.search_space.get_all_trivial_features()]

    def is_valid_hot_encoded_feature(self, hot_feature):
        deconcatted = self.deconcat_hot_encoding(hot_feature)

        def is_valid_one_hot_encoding(hot_enc):
            return np.sum(hot_enc) < 2

        return all([is_valid_one_hot_encoding(enc) for enc in deconcatted])


def merge_features(feature_hot_a: np.ndarray, feature_hot_b: np.ndarray):
    return np.minimum(feature_hot_a + feature_hot_b, 1)


def merge_many_features(list_of_hot_features):
    return np.minimum(np.sum(list_of_hot_features, axis=0), 1)


def features_are_equal(raw_a, raw_b):
    return np.array_equal(raw_a, raw_b)
