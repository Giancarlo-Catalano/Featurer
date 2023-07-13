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

    def get_empty_feature(self):
        return Feature([None]*self.dimensions)
    def get_single_value_feature(self, var, val):
        result = self.get_empty_feature()
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

    def merge_two_features(self, feature_a: Feature, feature_b: Feature):
        return Feature([from_a or from_b for (from_a, from_b) in zip(feature_a.values, feature_b.values)])

    def merge_two_features_safe(self, feature_a: Feature, feature_b: Feature):
        print(f"Merging {feature_a} and {feature_b}", end="")
        result:Feature = Feature([])
        # I'm sorry!!! but having 0 or None was really confusing..
        for from_a, from_b in zip(feature_a.values, feature_b.values):
            if from_a is None:
                if from_b is None:
                    result.values.append(None)
                else:
                    result.values.append(from_b)
            else:
                if from_b is None:
                    result.values.append(from_a)
                elif from_a != from_b:
                    print("...Failed")
                    return None
                else:
                    result.values.append(from_a)


        print(f" = {result}")
        return result

    def merge_features_safe(self, list_of_features):
        # assumes list_of_features has at least 2 elements
        result = self.merge_two_features_safe(list_of_features[0], list_of_features[1])
        for feature in list_of_features[2:]:
            if result is None:
                return None
            result = self.merge_two_features_safe(result, feature)
        if result is None:
            return None
        return result
    def merge_features(self, list_of_features):
        # assumes list_of_features has at least 2 elements
        result = self.merge_two_features(list_of_features[0], list_of_features[1])
        for feature in list_of_features[2:]:
            result = self.merge_two_features(result, feature)
        return result







