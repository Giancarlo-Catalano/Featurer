import itertools
from typing import Optional

import numpy as np
from bitarray import bitarray, frozenbitarray

from Version_E.Feature import Feature
import SearchSpace
import utils


def get_hot_encoded_feature(feature: Feature, search_space: SearchSpace):
    result = np.zeros(search_space.total_cardinality, dtype=float)

    def position_in_hot_encoded(var_index, val_index):
        return search_space.precomputed_offsets[var_index] + val_index

    for var, val in feature.to_var_val_pairs():
        index = position_in_hot_encoded(var, val)
        result[index] = 1.0
    return result

def hot_encode_candidate(candidate: SearchSpace.Candidate, search_space: SearchSpace)-> np.ndarray:
    return np.concatenate(
        [utils.one_hot_encoding(candidate.values[var], cardinality)
         for (var, cardinality) in enumerate(search_space.cardinalities)])

def hot_encode_candidate_population(population: list[SearchSpace.Candidate],
                                    search_space: SearchSpace.SearchSpace) -> np.ndarray:
    return np.array([hot_encode_candidate(c, search_space) for c in population])


def feature_from_hot_encoding(hot_encoded: np.ndarray, search_space: SearchSpace.SearchSpace) -> Feature:
    def deconcat_hot_encoding(hot_encoded_input: np.ndarray) -> list[np.ndarray]:
        return [hot_encoded_input[begin:end]
                for (begin, end) in itertools.pairwise(search_space.precomputed_offsets)]

    def from_hot_encoding(hot_encoded_int: np.ndarray) -> Optional[int]:
        for index, value in enumerate(hot_encoded_int):
            if value == 1.0:
                return index
        return None

    def tuple_to_feature(tuple_input: tuple) -> Feature:
        var_mask = bitarray(value is not None for value in tuple_input)
        val_mask = np.array([0 if value is None else value for value in tuple_input])

        return Feature(frozenbitarray(var_mask), val_mask)

    deconcatted = deconcat_hot_encoding(hot_encoded)
    values_tuple = tuple(from_hot_encoding(item) for item in deconcatted)
    return tuple_to_feature(values_tuple)

