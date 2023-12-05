import random
from typing import Iterable

import numpy as np
from bitarray import frozenbitarray

import SearchSpace
import utils
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem
from Version_E.Feature import Feature


class Permutation:

    shuffling: list[int]
    unshuffling: list[int]

    def __init__(self, amount_of_items: int):
        self.shuffling = random.sample(range(amount_of_items), amount_of_items)
        shuffled_pairs = sorted(enumerate(self.shuffling), key=utils.second)
        self.unshuffling, _ = utils.unzip(shuffled_pairs)

    def shuffle(self, items) -> list:
        assert(len(items) == len(self.shuffling))
        return [items[index] for index in self.shuffling]

    def unshuffle(self, items) -> list:
        assert (len(items) == len(self.shuffling))
        return [items[index] for index in self.unshuffling]


def shuffle_feature(feature: Feature, permutation: Permutation) -> Feature:
    new_mask = frozenbitarray(permutation.shuffle(feature.variable_mask))
    new_vals = np.array(permutation.shuffle(feature.values_mask))

    return Feature(new_mask, new_vals)

class ShuffleProblem(TestableCombinatorialProblem):
    original_problem: TestableCombinatorialProblem
    permutation: Permutation


    def shuffle_feature(self, feature: Feature) -> Feature:
        new_mask = frozenbitarray(self.permutation.shuffle(feature.variable_mask))
        new_vals = np.array(self.permutation.shuffle(feature.values_mask))

        return Feature(new_mask, new_vals)


    def unshuffle_feature(self, feature: Feature) -> Feature:
        new_mask = frozenbitarray(self.permutation.unshuffle(feature.variable_mask))
        new_vals = np.array(self.permutation.unshuffle(feature.values_mask))

        return Feature(new_mask, new_vals)


    def unshuffle_legacy_feature(self, feature: SearchSpace.UserFeature):
        feature_internal = Feature.from_legacy_feature(feature, self.search_space)
        unshuffled_feature = self.unshuffle_feature(feature_internal)
        return unshuffled_feature.to_legacy_feature()


    def shuffle_candidate(self, candidate: SearchSpace.Candidate) -> SearchSpace.Candidate:
        return SearchSpace.Candidate(self.permutation.shuffle(candidate.values))

    def unshuffle_candidate(self, candidate: SearchSpace.Candidate) -> SearchSpace.Candidate:
        return SearchSpace.Candidate(self.permutation.unshuffle(candidate.values))

    def __init__(self, original_problem: TestableCombinatorialProblem):
        self.original_problem = original_problem
        self.permutation = Permutation(self.original_problem.search_space.dimensions)

        original_search_space_cardinalities = self.original_problem.search_space.cardinalities
        shuffled_cardinalities = self.permutation.shuffle(original_search_space_cardinalities)
        super().__init__(SearchSpace.SearchSpace(shuffled_cardinalities))


    def __repr__(self):
        return f"Shuffle of {self.original_problem}"

    def long_repr(self):
        return f"Shuffle of {self.original_problem.long_repr()}"

    def get_complexity_of_feature(self, feature: SearchSpace.UserFeature):
        return self.original_problem.get_complexity_of_feature(self.unshuffle_legacy_feature(feature))

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        unshuffled_feature = self.unshuffle_candidate(candidate)
        return self.original_problem.score_of_candidate(unshuffled_feature)

    def feature_repr(self, feature):
        return f"{Feature.from_legacy_feature(feature, self.search_space)}"

    def get_ideal_features(self) -> list[Feature]:
        ideals = self.original_problem.get_ideal_features()
        return [self.shuffle_feature(ideal) for ideal in ideals]


