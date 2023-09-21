from typing import Optional

import SearchSpace
import numpy as np
from bitarray import bitarray, frozenbitarray

import utils


class Feature:
    variable_mask: frozenbitarray
    values_mask: np.ndarray

    def __init__(self, variable_mask: frozenbitarray, values_mask: np.ndarray):
        self.variable_mask = variable_mask
        self.values_mask = values_mask
        self.values_mask.setflags(write=False)

    @property
    def dimensions(self) -> int:
        return len(self.values_mask)

    def value_at(self, index: int) -> int:
        return self.values_mask[index]

    def to_var_val_pairs(self) -> list[(int, int)]:
        return [(var, self.value_at(var))
                for (var, is_used) in enumerate(self.variable_mask)
                if is_used]

    def to_candidate(self) -> SearchSpace.Candidate:
        values = [0] * self.dimensions
        for var, val in self.to_var_val_pairs():
            values[var] = val
        return SearchSpace.Candidate(values)

    @classmethod
    def overlap(cls, first, second) -> bool:
        overlap: frozenbitarray = first.variable_mask & second.variable_mask
        return overlap.any()

    @classmethod
    def merge_is_good(cls, first, second) -> bool:
        merged = first.variable_mask | second.variable_mask
        return (merged != first.variable_mask) and (merged != second.variable_mask)

    @classmethod
    def are_disjoint(cls, first, second) -> bool:
        return not cls.overlap(first, second)

    @classmethod
    def merge(cls, first, second):
        new_values = np.maximum(first.values_mask, second.values_mask)
        new_mask = first.variable_mask | second.variable_mask
        return cls(new_mask, new_values)

    @classmethod
    def empty_feature(cls, search_space: SearchSpace.SearchSpace):
        var_mask = bitarray(search_space.dimensions)
        var_mask.setall(0)
        val_mask = np.zeros(search_space.dimensions, dtype=int)

        return cls(frozenbitarray(var_mask), val_mask)

    @classmethod
    def from_trivial_feature(cls, var, val, search_space: SearchSpace.SearchSpace):
        var_mask = bitarray(search_space.dimensions)
        var_mask.setall(0)
        val_mask = np.zeros(search_space.dimensions, dtype=int)

        var_mask[var] = True
        val_mask[var] = val
        return cls(frozenbitarray(var_mask), val_mask)

    @classmethod
    def get_all_trivial_features(cls, search_space: SearchSpace.SearchSpace):
        return [Feature.from_trivial_feature(var, val, search_space)
                for var, val in search_space.get_all_var_val_pairs()]

    def __hash__(self):
        var_hash = self.variable_mask.__hash__()
        # NOTE: using
        # val_hash = hash(tuple(self.variable_mask))
        # also works well.
        val_hash = hash(np.bitwise_xor.reduce(self.variable_mask))
        return var_hash + val_hash

    def __eq__(self, other) -> bool:
        if self.variable_mask != other.variable_mask:
            return False

        for value_here, value_there, is_set in zip(self.values_mask, other.values_mask, self.variable_mask):
            if is_set and value_here != value_there:
                return False
        return True

        # return self.variable_mask == other.variable_mask and np.array_equal(self.values_mask, other.values_mask)

    def __repr__(self):
        result = ""
        for is_used, val in zip(self.variable_mask, self.values_mask):
            if is_used:
                result += f"{val}"
            else:
                result += "_"

        return result

    def to_legacy_feature(self) -> SearchSpace.UserFeature:
        return SearchSpace.UserFeature(self.to_var_val_pairs())

    @classmethod
    def from_candidate(cls, candidate: SearchSpace.Candidate):
        variable_mask = bitarray(len(candidate.values))
        variable_mask.setall(1)
        value_mask = np.array(candidate.values)
        return cls(frozenbitarray(variable_mask), value_mask)

    @classmethod
    def candidate_matrix_to_features(cls, candidate_matrix: np.ndarray, search_space: SearchSpace.SearchSpace):
        values_in_hot_encoding = np.array(utils.concat_lists([list(range(cardinality))
                                                              for cardinality in search_space.cardinalities]))

        value_matrix = np.array(candidate_matrix, dtype=int) * values_in_hot_encoding

        def get_values_for_variable(var_index: int):
            start, end = search_space.precomputed_offsets[var_index:(var_index + 2)]
            return np.max(value_matrix[:, start:end], axis=1)

        values_for_each_candidate = np.column_stack(tuple([get_values_for_variable(var)
                                                           for var in range(search_space.dimensions)]))

        variable_mask = bitarray(search_space.dimensions)
        variable_mask.setall(1)
        variable_mask = frozenbitarray(variable_mask)

        return [cls(variable_mask, row) for row in values_for_each_candidate]


    def with_value(self, var_index: int, val: Optional[int]):
        """returns a copy of itself, but the variable is changed to the supplied value"""
        """if the value is None, the cell becomes unset"""
        new_mask = bitarray(self.variable_mask)
        new_mask[var_index] = (val is not None)
        new_mask = frozenbitarray(new_mask)

        new_values = self.values_mask.copy()
        new_values[val] = 0 if val is None else val
        return Feature(new_mask, new_values)


    def get_generalisations(self) -> list:
        def get_decayed_masks(bitmask: frozenbitarray):
            result = []
            for index, is_set in enumerate(bitmask):
                if is_set:
                    new_decay = bitarray(self.variable_mask)
                    new_decay[index] ^= 1
                    result.append(frozenbitarray(new_decay))
            return result

        return [Feature(decayed_mask, self.values_mask) for decayed_mask in get_decayed_masks(self.variable_mask)]

    def get_specialisations(self, search_space: SearchSpace) -> list:
        def get_specialisation_pairs():
            result = []
            for var_index, is_used, cardinality in zip(range(search_space.dimensions), self.variable_mask,
                                                       search_space.cardinalities):
                if not is_used:
                    result.extend([(var_index, value) for value in range(cardinality)])

            return result

        def get_specialisation(var, val):
            new_mask = bitarray(self.variable_mask)
            new_mask[var] ^= 1
            new_mask = frozenbitarray(new_mask)
            new_vals = self.values_mask.copy()
            new_vals[var] = val
            return Feature(new_mask, new_vals)

        return [get_specialisation(var, val) for var, val in get_specialisation_pairs()]

    def get_variations(self, search_space: SearchSpace):
        """returns copies of itself, but one of the values has been changed"""

        def self_but_with_variation(var_index, val):
            result_mask = self.variable_mask
            value_mask = self.values_mask.copy()
            value_mask[var_index] = val
            return Feature(result_mask, value_mask)

        return [self_but_with_variation(var_index, alternative)
                for var_index in range(search_space.dimensions)
                for alternative in range(search_space.cardinalities[var_index])
                if alternative != self.values_mask[var_index]]
