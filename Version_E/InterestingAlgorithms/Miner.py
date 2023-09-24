import random

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

    def get_meaningful_features(self, amount_to_return: int) -> list[UserFeature]:
        mined_features = self.mine_features()
        kept_features_with_scores = self.feature_selector.keep_best_features(mined_features, amount_to_return)
        kept_features, scores = utils.unzip(kept_features_with_scores)
        return [feature.to_legacy_feature() for feature in kept_features]


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

    def branch_from_feature(self, feature: Feature) -> list[Feature]:
        raise Exception("An implementation of FeatureMiner does not implement branch_from_feature")

    def should_terminate(self, next_iteration: int):
        raise Exception("An implementation of FeatureMixer does not implement should_terminate")

    def make_layer(self, features: list[Feature]) -> Layer:
        return list(zip(features, self.feature_selector.get_scores(features)))

    def keep_top_features_and_make_layer(self, features: list[Feature], amount_to_keep: int) -> Layer:
        return self.feature_selector.keep_best_features(features, amount_to_keep)

    def choose_features_stochastically(self, previous_layer: Layer) -> set[Feature]:
        tournament_size = 30
        features_with_scores = previous_layer
        weights = utils.unzip(features_with_scores)[1]

        def tournament_select() -> Feature:
            extracted = random.choices(features_with_scores, weights, k=tournament_size)
            return max(extracted, key=utils.second)[0]

        selected = set()
        while len(selected) < self.amount_to_keep_in_each_layer:
            selected.add(tournament_select())

        return selected

    def select_features(self, previous_layer) -> set[Feature]:
        if len(previous_layer) <= self.amount_to_keep_in_each_layer:
            return set(utils.unzip(previous_layer)[0])

        if self.stochastic:
            return self.choose_features_stochastically(previous_layer)
        else:
            return set(utils.unzip(previous_layer)[0])

    def get_next_layer(self, prev_layer: Layer) -> Layer:
        selected_features: set[Feature] = self.select_features(prev_layer)

        modified_features = utils.concat_lists(self.branch_from_feature(feature) for feature in selected_features)
        culled_features = list(set(modified_features))
        return self.keep_top_features_and_make_layer(culled_features, self.amount_to_keep_in_each_layer)

    def mine_features(self) -> list[Feature]:
        initial_features = self.get_initial_features(self.feature_selector.ppi)
        initial_layer = self.make_layer(initial_features)
        layers: list[Layer] = [initial_layer]

        iteration = 0
        while True:
            iteration += 1
            print(f"We're at iteration {iteration}")
            if self.should_terminate(iteration):
                break

            layers.append(self.get_next_layer(layers[-1]))

        final_features = [feature
                          for layer in layers
                          for feature, score in layer]
        return final_features
