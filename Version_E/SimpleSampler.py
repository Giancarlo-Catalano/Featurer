import random
from copy import copy

import numpy as np

import SearchSpace
import utils
from Version_E.Feature import Feature


class SimpleSampler:
    search_space: SearchSpace.SearchSpace
    features_with_scores: list[(Feature, float)]


    def __init__(self, search_space: SearchSpace.SearchSpace, features_with_scores: list[(Feature, float)]):
        self.search_space = search_space
        self.features_with_scores = features_with_scores
        scores = utils.unzip(self.features_with_scores)[1]
        self.cumulative_distribution = np.cumsum(scores)

    def __repr__(self):
        return f"SimpleSampler(amount_of_features = {len(self.features_with_scores)}"

    def can_be_merged(self, feature_a: Feature, feature_b: Feature) -> bool:
        intersections = feature_a.variable_mask & feature_b.variable_mask
        if not any(intersections):
            return True

        for index, does_intersect in enumerate(intersections):
            if does_intersect and feature_a.value_at(index) != feature_b.value_at(index):
                return False

        return True


    def feature_is_complete(self, feature: Feature) -> bool:
        return all(feature.variable_mask)

    def sample_feature_unsafe(self) -> Feature:
        """ this is unsafe in the sense that the result might not be complete"""

        available_features = list(self.features_with_scores)

        def pick_a_feature() -> Feature:
            scores = utils.unzip(available_features)[1]
            indexes = list(range(len(available_features)))
            choice = random.choices(indexes, weights=scores, k=1)[0]

            picked_feature, _ = available_features.pop(choice)
            return picked_feature

        accumulated_feature = pick_a_feature()

        def can_still_sample():
            if len(available_features) < 1:
                return False

            weights = utils.unzip(available_features)[1]
            if np.sum(weights) == 0:
                return False
            return True

        while can_still_sample() and not self.feature_is_complete(accumulated_feature):
            feature_to_add = pick_a_feature()
            if self.can_be_merged(accumulated_feature, feature_to_add):
                accumulated_feature = Feature.merge(accumulated_feature, feature_to_add)

        return accumulated_feature  # it might be incomplete!!


    def fill_in_the_gaps(self, incomplete_feature: Feature):
        result = copy(incomplete_feature)
        for index, is_set in enumerate(incomplete_feature.variable_mask):
            if not is_set:
                options = list(range(self.search_space.cardinalities[index]))
                chosen_value = random.choice(options)
                result = result.with_value(index, chosen_value)
        return result

    def sample_candidate(self) -> SearchSpace.Candidate:
        produced_feature = self.sample_feature_unsafe()
        filled_feature = self.fill_in_the_gaps(produced_feature)
        return filled_feature.to_candidate()





