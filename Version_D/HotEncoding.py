import numpy as np
from Version_D.Feature import Feature
import SearchSpace


def get_hot_encoded_feature(feature: Feature, search_space: SearchSpace):
    result = np.zeros(search_space.total_cardinality, dtype=float)

    def position_in_hot_encoded(var_index, val_index):
        return search_space.precomputed_offsets[var_index] + val_index

    for var, val in feature.to_var_val_pairs():
        index = position_in_hot_encoded(var, val)
        result[index] = 1.0
    return result
