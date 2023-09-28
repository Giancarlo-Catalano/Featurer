import random

import SearchSpace
import BenchmarkProblems.CombinatorialProblem
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem
from typing import List, Optional

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

    important_features: list[SearchSpace.UserFeature]
    score_for_each_feature: list[int]


    def get_random_feature(self):
        value = random.randrange(2)
        start = random.randrange(self.amount_of_bits-self.size_of_features)
        which_vars = [start+offset for offset in range(self.size_of_features)]
        feature = SearchSpace.UserFeature([(var, value) for var in which_vars])
        return feature

    def features_are_disjoint(self, feature_a: SearchSpace.UserFeature, feature_b: SearchSpace.UserFeature):
            var_vals_a = set(feature_a.var_vals)
            var_vals_b = set(feature_b.var_vals)
            overlap = var_vals_a.intersection(var_vals_b)
            return len(overlap) == 0


    def generate_features(self) -> list[SearchSpace.UserFeature]:
        accumulator = set()
        def is_eligible(new_feature: SearchSpace.UserFeature):
            return all([self.features_are_disjoint(old_feature, new_feature) for old_feature in accumulator])

        attempts_until_next_reset = 1000
        while len(accumulator) < self.amount_of_features:
            new_feature = self.get_random_feature()
            if self.allow_overlaps or (not self.allow_overlaps and is_eligible(new_feature)):
                accumulator.add(new_feature)
            attempts_until_next_reset -= 1
            if attempts_until_next_reset < 1:
                accumulator = set()
                attempts_until_next_reset = 1000

        return list(accumulator)


    def generate_scores_for_features(self):
        amount_needed = len(self.important_features)
        scores = list(range(1, amount_needed+1))
        random.shuffle(scores)
        return scores

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

    def get_ideal_features(self) -> list[SearchSpace.UserFeature]:
        return self.important_features

    def long_repr(self):
        def repr_for_each_feature(feature: SearchSpace.UserFeature, value: int):
            return f"\t{self.feature_repr(feature)}, value = {value}"


        return ("Contains the following features:\n"+
                "\n".join([repr_for_each_feature(f, v)
                 for f, v in zip(self.important_features, self.score_for_each_feature)]))

    def get_complexity_of_feature(self, feature: SearchSpace.UserFeature):
        return super().get_area_of_smallest_bounding_box(feature) ** 2

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        return sum(score for feature, score in zip(self.important_features, self.score_for_each_feature)
                   if candidate.contains_feature(feature))

    def feature_repr(self, feature: SearchSpace.UserFeature):
        result_values = [None] * self.amount_of_bits
        for var, val in feature.var_vals:
            result_values[var] = val

        def repr_cell(value: Optional[int]):
            if value is None:
                return "_"
            else:
                return f"{value}"

        return " ".join(repr_cell(cell) for cell in result_values)



