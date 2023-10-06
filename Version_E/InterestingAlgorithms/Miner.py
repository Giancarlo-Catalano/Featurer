import math
import random
from typing import Iterable

import SearchSpace
import utils
from Version_E.Feature import Feature
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
import numpy as np

from SearchSpace import UserFeature

Score = float


class FeatureSelector:
    ppi: PrecomputedPopulationInformation
    criterion: MeasurableCriterion

    def __init__(self, ppi: PrecomputedPopulationInformation, criterion: MeasurableCriterion):
        self.ppi = ppi
        self.criterion = criterion

    def __repr__(self):
        return "FeatureSelector"

    def get_scores(self, features: list[Feature]) -> np.ndarray:
        pfi = PrecomputedFeatureInformation(self.ppi, features)
        return self.criterion.get_score_array(pfi)

    def keep_best_features(self, features: list[Feature], amount_to_keep: int) -> list[(Feature, Score)]:
        scores = self.get_scores(features)
        sorted_paired = sorted(zip(features, scores), key=utils.second, reverse=True)
        return sorted_paired[:amount_to_keep]


class FeatureMiner:
    feature_selector: FeatureSelector

    def __init__(self, feature_selector: FeatureSelector):
        self.feature_selector = feature_selector

    @property
    def search_space(self):
        return self.feature_selector.ppi.search_space

    def mine_features(self) -> list[Feature]:
        raise Exception("An implementation of FeatureMiner does not implement mine_features")

    def cull_subsets(self, features: list[Feature]) -> list[(Feature, Score)]:
        # TODO this is not working as intended, some subsets are still present at the end!!!
        features_with_scores = list(zip(features, self.feature_selector.get_scores(features)))
        features_with_scores.sort(key=utils.second, reverse=True)
        kept = []

        def consider_feature(feature: Feature, score: Score):
            for index, (other_feature, other_score) in enumerate(kept):
                if feature.is_subset_of(other_feature) or other_feature.is_subset_of(feature):
                    if score > other_score:
                        kept[index] = feature, score
                    return
            kept.append((feature, score))

        for feature, score in features_with_scores:
            consider_feature(feature, score)

        return kept

    def get_meaningful_features(self, amount_to_return: int, cull_subsets=False) -> list[Feature]:
        mined_features = self.mine_features()
        if cull_subsets:
            culled_features = self.cull_subsets(mined_features)
        else:
            culled_features = zip(mined_features, self.feature_selector.get_scores(mined_features))

        kept_features_with_scores = sorted(culled_features, key=utils.second, reverse=True)[:amount_to_return]
        return [feature
                for feature, score in kept_features_with_scores]


Layer = list[(Feature, Score)]





class LayeredFeatureMiner(FeatureMiner):
    amount_to_keep_in_each_layer: int
    stochastic: bool

    def __init__(self, selector: FeatureSelector, amount_to_keep_in_each_layer: int, stochastic: bool):
        super().__init__(selector)
        self.amount_to_keep_in_each_layer = amount_to_keep_in_each_layer
        self.stochastic = stochastic

    def get_initial_features(self, ppi: PrecomputedPopulationInformation) -> list[Feature]:
        raise Exception("An implementation of FeatureMiner does not implement get_initial_layer")

    def modifications_of_feature(self, feature: Feature) -> list[Feature]:
        raise Exception("An implementation of FeatureMiner does not implement branch_from_feature")

    def should_terminate(self, next_iteration: int):
        raise Exception("An implementation of FeatureMixer does not implement should_terminate")

    def make_layer(self, features: list[Feature]) -> Layer:
        return list(zip(features, self.feature_selector.get_scores(features)))

    def keep_top_features_and_make_layer(self, features: list[Feature], amount_to_keep: int) -> Layer:
        return self.feature_selector.keep_best_features(features, amount_to_keep)

    def select_features_stochastically(self, previous_layer: Layer, amount_to_return: int) -> list[Feature]:
        """
        This function selects features using tournament selection, returning a distinct set of features
        :param previous_layer: layer to select from
        :param amount_to_return: Res Ipsa Loquitur
        :return:
        """
        features, weights = utils.unzip(previous_layer)
        selected = random.choices(features, weights=weights, k=amount_to_return)

        return list(selected)

    def select_features_heuristically(self, previous_layer: Layer, amount_to_return: int) -> list[Feature]:
        """Select features using truncation selection"""
        # ensure previous layer is sorted
        previous_layer.sort(key=utils.second, reverse=True)
        return [feature for feature, _ in previous_layer[:amount_to_return]]

    def select_features(self, previous_layer) -> list[Feature]:
        """
        This function selects a portion of the population, using a different method based on internal parameters
        :param previous_layer: the layer to select from
        :return: returns a set of features
        """

        proportion = 0.5
        amount_to_return = math.ceil(len(previous_layer) * proportion)
        if len(previous_layer) <= amount_to_return:
            return utils.unzip(previous_layer)[0]

        if self.stochastic:
            return self.select_features_stochastically(previous_layer, amount_to_return)
        else:
            return self.select_features_heuristically(previous_layer, amount_to_return)

    def mine_features(self) -> list[Feature]:

        def truncate_and_make_layer(features: list[Feature]) -> Layer:
            return self.feature_selector.keep_best_features(features, self.amount_to_keep_in_each_layer)

        def get_initial_layer():
            initial_features = self.get_initial_features(self.feature_selector.ppi)
            return truncate_and_make_layer(initial_features)

        def get_next_layer(prev_layer: Layer) -> Layer:
            selected_features: list[Feature] = self.select_features(prev_layer)

            modified_features = [modified_feature
                                 for feature in selected_features
                                 for modified_feature in self.modifications_of_feature(feature)]
            modified_features = list(set(modified_features)) # to remove duplicates
            return truncate_and_make_layer(modified_features)

        layers: list[Layer] = [get_initial_layer()]

        iteration = 0
        while True:
            iteration += 1
            # print(f"We're at iteration {iteration}")
            if self.should_terminate(iteration):
                break

            layers.append(get_next_layer(layers[-1]))

        final_features = [feature
                          for layer in layers
                          for feature, _ in layer]
        return final_features
