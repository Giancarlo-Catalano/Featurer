import numpy as np
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
