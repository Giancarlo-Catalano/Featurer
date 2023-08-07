import copy
import itertools
import math
import random
import utils
import numpy as np


def value_to_string(val):
    if val is None:
        return "*"
    elif math.isnan(val):
        return "#"
    else:
        return str(val)


class Feature:
    # a list of pairs (var, val)
    def __init__(self, var_vals):
        self.var_vals = var_vals

    def __repr__(self):
        return "<" + (", ".join([f"[{var}]={val}" for var, val in self.var_vals])) + ">"

    def __hash__(self):
        return tuple(self.var_vals).__hash__()

    @classmethod
    def empty_feature(cls):
        return cls([])

    @classmethod
    def trivial_feature(cls, var, val):
        return cls([(var, val)])


class Candidate:
    # a wrapper for a tuple
    def __init__(self, values):
        self.values = values

    def __repr__(self):
        return "<" + (" ".join([value_to_string(val) for val in self.values])) + ">"

    def __eq__(self, other):
        return self.values == other.values

    def __hash__(self):
        return self.values.__hash__()


class SearchSpace:
    # cardinalities: list[int]
    # precomputed_offsets: list[int] #used to convert into one-hot-encodings

    def __init__(self, cardinalities):
        self.cardinalities = cardinalities
        self.precomputed_offsets = utils.cumulative_sum(cardinalities)
        self.amount_of_trivial_features = sum(self.cardinalities)
        self.dimensions = len(self.cardinalities)

    @property
    def total_cardinality(self):
        return sum(self.cardinalities)  # there's other ways, but this is more change resistant

    def get_random_candidate(self):
        return Candidate(tuple((random.randrange(card) for card in self.cardinalities)))

    def get_all_var_val_pairs(self):
        return [(var, val) for var in range(self.dimensions)
                for val in range(self.cardinalities[var])]

    def get_all_trivial_features(self):
        all_var_val_pairs = self.get_all_var_val_pairs()

        return [Feature.trivial_feature(var, val)
                for (var, val) in all_var_val_pairs]

    def probability_of_feature_in_uniform(self, combinatorial_feature: Feature):
        result = 1
        for var, _ in combinatorial_feature.var_vals:
            result /= self.cardinalities[var]
        return result

    def __repr__(self):
        return f"SearchSpace{self.cardinalities}"


def merge_two_features(feature_a, feature_b):
    return Feature(feature_a.var_vals + feature_b.var_vals)

