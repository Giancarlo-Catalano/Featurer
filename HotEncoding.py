import utils
import numpy as np
import SearchSpace

def candidate_contains_feature(candidate, feature_hot):
    # it's equivalent to NOT(ANY(NOT(C)&F))
    # which can be calculated as 0==SUM((1-C)&F)
    return 0 == np.sum((1 - candidate) & feature_hot)

def feature_is_valid_in_search_space(feature_hot, search_space: SearchSpace):
    def sum_between(start_index, end_index):
        return np.sum(feature_hot[start_index:end_index])

    return all([sum_between(begin, end) < 2
                for (begin, end) in utils.adjacent_pairs(search_space.precomputed_offsets)])

class HotEncoder:
    search_space: SearchSpace
    def __init__(self, search_space):
        self.search_space = search_space

    @property
    def empty_feature(self): # TO TEST
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
        decoded_individually = [utils.from_hot_encoding(hot) for hot in deconcatted]
        return tuple(decoded_individually)

    def candidate_from_hot_encoding(self, hot_encoded):
        return SearchSpace.Candidate(self.decode_tuple(hot_encoded))

    def feature_from_hot_encoding(self, hot_encoded):
        return SearchSpace.Feature(self.decode_tuple(hot_encoded))

    def to_hot_encoded_matrix(self, population):
        return np.array([self.to_hot_encoding(c) for c in population])


    def get_hot_encoded_trivial_features(self):
        return [self.to_hot_encoding(feature) for feature in self.search_space.get_all_trivial_features()]

    def is_valid_hot_encoded_feature(self, hot_feature):
        deconcatted = self.deconcat_hot_encoding(hot_feature)
        def is_valid_one_hot_encoding(hot_enc):
            return np.sum(hot_enc) < 2
        return all([is_valid_one_hot_encoding(enc) for enc in deconcatted])


def merge_features(feature_hot_a, feature_hot_b):
    return np.minimum(feature_hot_a+feature_hot_b, 1)


def merge_many_features(list_of_hot_features):
    return np.minimum(np.sum(list_of_hot_features, axis=0), 1)


def features_are_equal(raw_a, raw_b):
    return np.array_equal(raw_a, raw_b)


