import itertools
from typing import Iterable

import numpy as np

import SearchSpace
import utils
from Version_E import HotEncoding
from Version_E.Feature import Feature
from Version_E.MeasurableCriterion.CriterionUtilities import PPICachedData
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.MeasurableCriterion.Robustness import get_fuzzy_match_matrix, get_mean_of_fuzzy_match_matrix
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation

FlatArray = np.ndarray
Matrix = np.ndarray
ColumnVector = np.ndarray

ValuePosition = int  # the index
Fitness = float

InteractionScore = float
InteractionTable = Matrix

HotEncodedFeature = FlatArray


def flatten_var_val(var: int, val: int, search_space: SearchSpace.SearchSpace) -> int:
    return search_space.precomputed_offsets[var] + val

def get_normalised_fitnesses(ppi: PrecomputedPopulationInformation) -> FlatArray:
    in_zero_one = utils.remap_array_in_zero_one(ppi.fitness_array)
    with_sum_one = in_zero_one / sum(in_zero_one)
    return with_sum_one

def mutual_information(p1X: float, pX1: float, p11: float) -> float:
    denominator = p1X * pX1
    if denominator < 1e-6:
        return 0  # panic
    if p11 < 1e-6:
        return 0  # panic, a bit less. This makes sense!

    return p11 * np.log(p11 / denominator)

class Interaction(MeasurableCriterion):
    cached_normalised_fitnesses: PPICachedData
    cached_pX1s: PPICachedData

    def get_proportions_of_features(self, hot_encoded_features: Iterable[HotEncodedFeature],
                                    ppi: PrecomputedPopulationInformation) -> FlatArray:
        normalised_fitnesses: FlatArray = self.cached_normalised_fitnesses.get_data_for_ppi(ppi)
        temp_pfi = PrecomputedFeatureInformation.get_from_hot_encoded_features(ppi, np.array(hot_encoded_features))
        proportions = utils.weighted_sum_of_rows(temp_pfi.feature_presence_matrix, normalised_fitnesses)
        return proportions

    def __init__(self):
        self.cached_normalised_fitnesses = PPICachedData(get_normalised_fitnesses)

        def get_P1Xs(ppi: PrecomputedPopulationInformation) -> FlatArray:
            amount_of_values = ppi.search_space.total_cardinality
            trivial_features = np.identity(amount_of_values, dtype=float)
            return self.get_proportions_of_features(trivial_features, ppi)

        self.cached_pX1s = PPICachedData(get_P1Xs)

    def linkage_scores_for_feature(self, feature: HotEncodedFeature,
                                   ppi: PrecomputedPopulationInformation,
                                   p11: float) -> list[float]:
        present_vals = [val_pos for val_pos, is_used in enumerate(feature) if is_used]

        if len(present_vals) == 0:
            return []

        def without_that_val(val_pos: int) -> HotEncodedFeature:
            without = np.array(feature)
            without[val_pos] = 0.0
            return without

        f1Xs = list(map(without_that_val, present_vals))

        p1Xs: FlatArray = self.get_proportions_of_features(f1Xs, ppi)
        all_pX1s: FlatArray = self.cached_pX1s.get_data_for_ppi(ppi)
        pX1s = [pX1 for pX1, is_used in zip(all_pX1s, feature) if is_used]

        result = [mutual_information(p1X, pX1, p11) for p1X, pX1 in zip(p1Xs, pX1s)]

        # if any(value < 0 for value in result):
        #     print(f"The feature {HotEncoding.feature_from_hot_encoding(feature, ppi.search_space)} has {result}")

        return result

    def faster_linkage_score_for_feature(self, feature: HotEncodedFeature,
                                         ppi: PrecomputedPopulationInformation,
                                         p11: float) -> float:
        present_vals = [val_pos for val_pos, is_used in enumerate(feature) if is_used]

        if len(present_vals) == 0:
            return 0

        if p11 < 1e-6:
            return 0

        def without_that_val(val_pos: int) -> HotEncodedFeature:
            without = np.array(feature)
            without[val_pos] = 0.0
            return without

        f1Xs = list(map(without_that_val, present_vals))

        p1Xs: FlatArray = self.get_proportions_of_features(f1Xs, ppi)
        all_pX1s: FlatArray = self.cached_pX1s.get_data_for_ppi(ppi)
        pX1s = all_pX1s[
            np.array(feature, dtype=bool)]  # feature is used as the predicate to select the marginal probabilities

        max_pX1_p1X = np.max(pX1s * p1Xs)
        return p11 * np.log2(p11 / max_pX1_p1X)

    def get_weakest_list_for_feature(self, feature: HotEncodedFeature,
                                     ppi: PrecomputedPopulationInformation,
                                     p11: float) -> float:
        if np.sum(feature) < 1:
            return 0

        linkage_scores = self.linkage_scores_for_feature(feature, ppi, p11)
        return np.min(linkage_scores)  # IMPORTANT

    def __repr__(self):
        return f"WeakestLink"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        ppi = pfi.precomputed_population_information
        normalised_fitnesses: FlatArray = self.cached_normalised_fitnesses.get_data_for_ppi(ppi)
        p11s = utils.weighted_sum_of_rows(pfi.feature_presence_matrix, normalised_fitnesses)

        hot_encoded_features = pfi.feature_matrix.T

        scores = np.array([self.faster_linkage_score_for_feature(feature, ppi, p11)
                           for feature, p11 in zip(hot_encoded_features, p11s)])

        return scores

    def describe_score(self, given_score) -> str:
        return f"Weakest_Link = {given_score:.2f}"
