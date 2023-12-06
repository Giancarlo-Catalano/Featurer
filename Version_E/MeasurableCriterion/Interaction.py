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

        linkage_scores = self.faster_linkage_score_for_feature(feature, ppi, p11)
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


class SimpleInteraction(MeasurableCriterion):
    cached_pX1s: PPICachedData

    def get_P1Xs(self, ppi: PrecomputedPopulationInformation) -> FlatArray:
        return np.mean(ppi.candidate_matrix, axis=0)

    def get_p1Xs_for_feature(self, feature_column_index: int,
                             pfi: PrecomputedFeatureInformation) -> (float, np.ndarray):
        errors_for_feature = pfi.feature_presence_error_matrix[:, feature_column_index]
        less_than_two_errors = errors_for_feature < 2
        rows_with_less_than_two_errors = pfi.candidate_matrix[less_than_two_errors]
        errors_filtered = errors_for_feature[less_than_two_errors]

        one_error_matches = rows_with_less_than_two_errors[errors_filtered == 1]

        amount_of_exact_matches = np.sum(errors_filtered == 0)
        feature = pfi.feature_matrix[:, feature_column_index]
        individual_errors = (1 - one_error_matches) * feature

        amount_of_less_than_2_matches, _ = rows_with_less_than_two_errors.shape
        matches_for_value = np.array([np.sum(row)
                                      for row in itertools.compress(individual_errors.T, feature)])

        p11 = amount_of_exact_matches / pfi.sample_size
        p1Xs = (matches_for_value+amount_of_exact_matches) / pfi.sample_size

        return p11, p1Xs

    def __init__(self):
        self.cached_pX1s = PPICachedData(self.get_P1Xs)

    def weakest_link_for_feature(self, feature_column_index: int,
                                         pfi: PrecomputedFeatureInformation) -> float:
        all_pX1s: FlatArray = self.cached_pX1s.get_data_for_ppi(pfi.precomputed_population_information)

        feature: HotEncodedFeature = pfi.feature_matrix[:, feature_column_index]
        if not any(feature):
            return 0
        pX1s = all_pX1s[feature == 1]
        p11, p1Xs = self.get_p1Xs_for_feature(feature_column_index, pfi)

        max_pX1_p1X = np.max(pX1s * p1Xs)
        return p11 * np.log2(p11 / max_pX1_p1X)

    def __repr__(self):
        return f"WeakestLink(P)"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        scores = np.array([self.weakest_link_for_feature(feature_index, pfi)
                           for feature_index in range(pfi.amount_of_features)])

        return scores

    def describe_score(self, given_score) -> str:
        return f"Weakest_Link(P) = {given_score:.2f}"
