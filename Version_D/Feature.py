import SearchSpace
import numpy as np
from bitarray import bitarray, frozenbitarray

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
        for var, val in self.get_used_variables():
            values[var] = val
        return SearchSpace.Candidate(values)

    @classmethod
    def are_disjoint(cls, first, second) -> bool:
        overlap: frozenbitarray = first.variable_mask & second.variable_mask
        return not (overlap.any())
    @classmethod
    def merge(cls, first, second):
        new_values = np.maximum(first.values_mask, second.values_mask)
        new_mask = first.variable_mask | second.variable_mask
        return cls(new_mask, new_values)

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
        return self.variable_mask == other.variable_mask and np.array_equal(self.values_mask, other.values_mask)

    def __repr__(self):
        result = ""
        for is_used, val in zip(self.variable_mask, self.values_mask):
            if is_used:
                result += f"{val}"
            else:
                result += "_"

        return result