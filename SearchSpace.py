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
    # a wrapper for a tuple of (optional) values,
    def __init__(self, values):
        self.values = values
    def __repr__(self):
        return "["+(" ".join([value_to_string(val) for val in self.values]))+"]"

    def __hash__(self):
        return self.values.__hash__()


    def __eq__(self, other):
        return self.values == other.values
class Candidate:
    # a wrapper for a tuple
    def __init__(self, values):
        self.values = values

    def __repr__(self):
        return "{"+(" ".join([value_to_string(val) for val in self.values]))+"}"


    def __eq__(self, other):
        return self.values == other.values

    def __hash__(self):
        return self.values.__hash__()

class SearchSpace:
    #cardinalities: list[int]
    #precomputed_offsets: list[int] #used to convert into one-hot-encodings

    def __init__(self, cardinalities):
        self.cardinalities = cardinalities
        self.precomputed_offsets = utils.cumulative_sum(cardinalities)
        self.amount_of_trivial_features = sum(self.cardinalities)
        self.dimensions = len(self.cardinalities)

    @property
    def total_cardinality(self):
        return sum(self.cardinalities) #there's other ways, but this is more change resistant

    def get_random_candidate(self):
        return Candidate(tuple((random.randrange(card) for card in self.cardinalities)))

    def get_single_value_feature(self, var, val):
        result = Feature([None]*self.dimensions)
        result.values[var] = val
        return result

    def get_all_trivial_features(self):
        all_var_val_pairs =  [(var, val) for var in range(self.dimensions)
                                         for val in range(self.cardinalities[var])]

        return [self.get_single_value_feature(var, val)
                    for (var, val) in all_var_val_pairs]


    def probability_of_feature_in_uniform(self, combinatorial_feature: Feature):
        result = 1
        for (val, card) in zip(combinatorial_feature.values, self.cardinalities):
            if val is not None:
                result /= card
        return result

    def __repr__(self):
        return f"SearchSpace{self.cardinalities}"






