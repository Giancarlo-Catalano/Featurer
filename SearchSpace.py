import math
import random
from typing import Optional, Iterable

import utils


def value_to_string(val):
    if val is None:
        return "*"
    elif math.isnan(val):
        return "#"
    else:
        return str(val)


class UserFeature:
    # a list of pairs (var, val)
    var_vals: list[(int, int)]

    def __init__(self, var_vals):
        self.var_vals = sorted(var_vals, key=utils.first)

    def __repr__(self):
        return "<" + (", ".join([f"[{var}]={val}" for var, val in self.var_vals])) + ">"

    def __hash__(self):
        return tuple(self.var_vals).__hash__()

    def __eq__(self, other):
        return self.var_vals == other.var_vals

    @classmethod
    def empty_feature(cls):
        return cls([])

    @classmethod
    def trivial_feature(cls, var, val):
        return cls([(var, val)])


class Candidate:
    # a wrapper for a tuple
    values: tuple[int]

    def __init__(self, values: Iterable[int]):
        self.values = tuple(values)

    def __repr__(self):
        return "<" + (" ".join([value_to_string(val) for val in self.values])) + ">"

    def __eq__(self, other):
        return self.values == other.values

    def __hash__(self):
        return self.values.__hash__()

    def as_feature(self):
        return UserFeature([(i, v) for i, v in enumerate(self.values) if v is not None])

    def contains_feature(self, feature: UserFeature):
        def contains_var_val(var, val):
            return self.values[var] == val

        return all(contains_var_val(var, val) for var, val in feature.var_vals)



class SearchSpace:
    cardinalities: tuple[int]
    precomputed_offsets: list[int]
    amount_of_trivial_features: int
    dimensions: int

    # precomputed_offsets: list[int] #used to convert into one-hot-encodings

    def __init__(self, cardinalities: Iterable[int]):
        self.cardinalities = tuple(cardinalities)
        self.precomputed_offsets = utils.cumulative_sum(self.cardinalities)
        self.amount_of_trivial_features = sum(self.cardinalities)
        self.dimensions = len(self.cardinalities)

    @property
    def total_cardinality(self):
        return sum(self.cardinalities)  # there's other ways, but this is more change resistant

    @property
    def average_cardinality(self):
        return self.total_cardinality / self.dimensions

    def get_random_candidate(self):
        return Candidate(tuple((random.randrange(card) for card in self.cardinalities)))

    def get_all_var_val_pairs(self):
        return [(var, val) for var in range(self.dimensions)
                for val in range(self.cardinalities[var])]

    def get_all_trivial_features(self):
        all_var_val_pairs = self.get_all_var_val_pairs()

        return [UserFeature.trivial_feature(var, val)
                for (var, val) in all_var_val_pairs]

    def probability_of_feature_in_uniform(self, combinatorial_feature: UserFeature):
        result = 1
        for var, _ in combinatorial_feature.var_vals:
            result /= self.cardinalities[var]
        return result

    def __repr__(self):
        return f"SearchSpace{self.cardinalities}"

    def feature_is_complete(self, feature: UserFeature):
        """returns true when the feature has all the variables set"""
        used_vars = [False] * self.dimensions
        for (var, _) in feature.var_vals:
            used_vars[var] = True
        return all(used_vars)

    def feature_is_valid(self, feature: UserFeature):
        value_for_each_var = [None] * self.dimensions
        for (var, val) in feature.var_vals:
            if value_for_each_var[var] is None:
                value_for_each_var[var] = val
            else:
                return False
        return True

    def feature_to_candidate(self, feature: UserFeature) -> Candidate:
        result_list = [None] * self.dimensions
        for var, val in feature.var_vals:
            result_list[var] = val
        return Candidate(tuple(result_list))

    def position_in_hot_encoded(self, var: int, val: int) -> int:
        return self.precomputed_offsets[var]+val

def merge_two_features(feature_a, feature_b) -> UserFeature:
    def remove_duplicates(input_list: list):
        return list(set(input_list))

    return UserFeature(remove_duplicates(feature_a.var_vals + feature_b.var_vals))


def merge_two_candidates(candidate_a: Candidate, candidate_b: Candidate) -> Candidate:
    return Candidate(candidate_a.values + candidate_b.values)


def merge_two_spaces(space_a: SearchSpace, space_b: SearchSpace) -> SearchSpace:
    return SearchSpace(space_a.cardinalities + space_b.cardinalities)


def merge_many_spaces(spaces: Iterable[SearchSpace]) -> SearchSpace:
    return SearchSpace(utils.concat_tuples(space.cardinalities for space in spaces))
