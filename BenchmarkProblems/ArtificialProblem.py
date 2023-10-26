import itertools
import random

import numpy as np
from bitarray import bitarray, frozenbitarray

import SearchSpace
import BenchmarkProblems.CombinatorialProblem
import utils
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem
from typing import List, Optional, Iterable

from Version_E.Feature import Feature


class ArtificialProblem(TestableCombinatorialProblem):
    """In this problem, the fitness function is given by the presence of some features"""
    """ To avoid issues, these features are guaranteed to be non-overlapping, meaning that all features are disjoint"""
    """ In other terms, no particular var-val combination is present in more than one feature. """
    """ BUT NOTE: the same var might be present in multiple features, just with different values"""

    amount_of_bits: int

    amount_of_features: int
    size_of_features: int
    allow_overlaps: bool

    important_features: list[Feature]
    score_for_each_feature: list[int]

    def get_random_feature(self) -> Feature:
        value = random.randrange(2)
        start = random.randrange(self.amount_of_bits - self.size_of_features)

        variable_mask = bitarray(self.search_space.dimensions)
        variable_mask.setall(0)
        value_mask = np.zeros(self.search_space.dimensions, dtype=int)
        variable_mask[start:(start + self.size_of_features)] = 1
        value_mask[start:(start + self.size_of_features)] = value

        return Feature(frozenbitarray(variable_mask), value_mask)

    def features_are_separate(self, feature_a: Feature, feature_b: Feature):
        def get_hitbox(b: bitarray) -> bitarray:
            distance = 1
            return (b << distance) | (b >> distance)

        def value_in_feature(f: Feature) -> int:
            index_of_first_set_value = f.variable_mask.index(1)
            return f.values_mask[index_of_first_set_value]

        hitbox_overlap = get_hitbox(feature_a.variable_mask) & get_hitbox(feature_b.variable_mask)
        if hitbox_overlap.any():
            return value_in_feature(feature_a) != value_in_feature(feature_b)
        else:
            return True

    def generate_features(self) -> list[Feature]:

        def generate_random_group() -> list[Feature]:
            return [self.get_random_feature() for _ in range(self.amount_of_features)]

        def feature_group_is_usable(group: Iterable[Feature]) -> bool:
            return all([self.features_are_separate(a, b) for a, b in itertools.combinations(group, 2)])

        while True:
            tentative_group = generate_random_group()
            if feature_group_is_usable(tentative_group):
                return tentative_group

    def generate_scores_for_features(self):
        amount_needed = len(self.important_features)
        scores = list(range(1, amount_needed + 1))
        random.shuffle(scores)
        return [1 for _ in self.important_features]  # scores

    def __init__(self, amount_of_bits, amount_of_features, size_of_partials, allow_overlaps):
        self.amount_of_bits = amount_of_bits
        self.amount_of_features = amount_of_features
        self.size_of_features = size_of_partials
        super().__init__(SearchSpace.SearchSpace([2] * self.amount_of_bits))
        self.allow_overlaps = allow_overlaps
        self.important_features = self.generate_features()
        self.score_for_each_feature = self.generate_scores_for_features()

    def __repr__(self):
        return (f"ArtificialProblem(bits={self.amount_of_bits}, "
                f"amount_of_features = {self.amount_of_features},"
                f"size_of_features = {self.size_of_features},"
                f"allow_overlap = {self.allow_overlaps}")

    def get_ideal_features(self) -> list[Feature]:
        return self.important_features

    def long_repr(self):
        features_with_weight = sorted(zip(self.important_features, self.score_for_each_feature),
                                      key=utils.second,
                                      reverse=True)

        return ("Contains the following features:\n" +
                "\n".join([f"\t{f}, value = {v}"
                           for f, v in features_with_weight]))

    def get_complexity_of_feature(self, feature: SearchSpace.UserFeature):
        return super().amount_of_set_values_in_feature(feature)

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        def contains_fast_feature(feature: Feature) -> bool:
            for val_here, val_there, is_used in zip(candidate.values, feature.values_mask, feature.variable_mask):
                if is_used and (val_here != val_there):
                    return False
            return True

        return sum(score for feature, score in zip(self.important_features, self.score_for_each_feature)
                   if contains_fast_feature(feature))

    def feature_repr(self, feature: SearchSpace.UserFeature):
        return Feature.from_legacy_feature(feature, self.search_space).__repr__()
