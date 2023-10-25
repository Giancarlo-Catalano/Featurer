import math
import random
import time
from typing import Iterable

import numpy as np

import utils
from Version_E import HotEncoding
from Version_E.Feature import Feature
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation

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
        if len(features) == 0:
            raise Exception("Not enough features to give a relative score")
        pfi = PrecomputedFeatureInformation(self.ppi, features)
        return self.criterion.get_score_array(pfi)

    def keep_best_features(self, features: Iterable[Feature], amount_to_keep: int) -> list[(Feature, Score)]:
        features_list = list(features)
        scores = self.get_scores(features_list)
        sorted_paired = sorted(zip(features_list, scores), key=utils.second, reverse=True)
        return sorted_paired[:amount_to_keep]

    def select_features_stochastically(self, features_with_scores: list[(Feature, Score)],
                                       amount_to_return: int) -> list[Feature]:
        """
        This function selects features using tournament selection, returning a distinct set of features
        :param features_with_scores: layer to select from
        :param amount_to_return: Res Ipsa Loquitur
        :return:
        """
        features, weights = utils.unzip(features_with_scores)
        random.seed(int(time.time()))  # to prevent predictable randomness
        selected = random.choices(features, weights=weights, k=amount_to_return)

        return list(selected)

    def select_features_heuristically(self, features_with_scores: list[(Feature, Score)], amount_to_return: int) -> \
    list[Feature]:
        """Select features using truncation selection"""
        # ensure previous layer is sorted
        features_with_scores.sort(key=utils.second, reverse=True)
        return [feature for feature, _ in features_with_scores[:amount_to_return]]


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



