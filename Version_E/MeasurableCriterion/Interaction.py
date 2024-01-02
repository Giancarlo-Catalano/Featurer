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


class SlowInteraction(MeasurableCriterion):
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


class Interaction(MeasurableCriterion):
    """ this uses both the proportion and the average"""
    cached_pX1s: PPICachedData
    cached_normalised_fitnesses: PPICachedData

    def get_P1Xs(self, ppi: PrecomputedPopulationInformation) -> FlatArray:
        normalised_fitnesses = self.cached_normalised_fitnesses.get_data_for_ppi(ppi)

        def get_P1X_for_column(val_column: int) -> float:
            where_val_is_used = ppi.candidate_matrix[:, val_column] == 1
            return np.sum(normalised_fitnesses, where=where_val_is_used)

        amount_of_values = ppi.search_space.total_cardinality
        return np.array([get_P1X_for_column(val_column) for val_column in range(amount_of_values)])

    def get_p1Xs_for_feature_adjusted_by_fitness(self, feature_column_index: int,
                                                 pfi: PrecomputedFeatureInformation,
                                                 normalised_fitnesses: FlatArray) -> (float, np.ndarray):
        errors_for_feature = pfi.feature_presence_error_matrix[:, feature_column_index]
        less_than_two_errors = errors_for_feature < 2
        feature = pfi.feature_matrix[:, feature_column_index]
        variable_is_used = feature == 1

        rows_with_less_than_two_errors = pfi.candidate_matrix[less_than_two_errors][:, variable_is_used]
        fitnesses_with_less_than_two_errors = normalised_fitnesses[less_than_two_errors]
        error_counts = errors_for_feature[less_than_two_errors]

        with_one_error = error_counts == 1
        candidates_with_one_error = rows_with_less_than_two_errors[with_one_error]
        fitnesses_with_one_error = fitnesses_with_less_than_two_errors[with_one_error]

        fitnesses_with_zero_errors = fitnesses_with_less_than_two_errors[error_counts == 0]

        individual_errors = (1 - candidates_with_one_error) * feature[variable_is_used]
        fitnesses_where_no_errors = individual_errors * fitnesses_with_one_error.reshape((-1, 1))

        p11 = np.sum(fitnesses_with_zero_errors)
        p1Xs = np.sum(fitnesses_where_no_errors, axis=0) + p11

        return p11, p1Xs

    def __init__(self):
        self.cached_pX1s = PPICachedData(self.get_P1Xs)
        self.cached_normalised_fitnesses = PPICachedData(get_normalised_fitnesses)

    def weakest_link_for_feature(self, feature_column_index: int,
                                 pfi: PrecomputedFeatureInformation) -> float:
        feature: HotEncodedFeature = pfi.feature_matrix[:, feature_column_index]
        if not any(feature):
            return 0
        all_pX1s: FlatArray = self.cached_pX1s.get_data_for_ppi(pfi.precomputed_population_information)
        normalised_fitnesses: FlatArray = self.cached_normalised_fitnesses.get_data_for_ppi(
            pfi.precomputed_population_information)

        pX1s = all_pX1s[feature == 1]
        p11, p1Xs = self.get_p1Xs_for_feature_adjusted_by_fitness(feature_column_index, pfi, normalised_fitnesses)

        if p11 == 0:
            return 0


        max_pX1_p1X = np.max(pX1s * p1Xs)
        result = p11 * np.log2(p11 / max_pX1_p1X)

        return result

    def __repr__(self):
        return f"WeakestLink(P)"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        scores = np.array([self.weakest_link_for_feature(feature_index, pfi)
                           for feature_index in range(pfi.amount_of_features)])

        return scores

    def describe_score(self, given_score) -> str:
        return f"Weakest_Link(P) = {given_score:.2f}"


class Artefact(MeasurableCriterion):
    cached_marginals: PPICachedData

    def get_marginal_probabilities(self, ppi: PrecomputedPopulationInformation) -> FlatArray:
        return np.sum(ppi.candidate_matrix, axis=0) / ppi.sample_size

    def get_observed_quantities(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return pfi.feature_presence_matrix.sum(axis=0)

    def get_expected_quantities(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        marginal_probabilities: np.ndarray = self.cached_marginals.get_data_for_ppi(
            pfi.precomputed_population_information)

        def get_individual_expected(feature: HotEncodedFeature):
            return np.product(marginal_probabilities[feature == 1]) * pfi.sample_size

        return np.array([get_individual_expected(feature) for feature in pfi.feature_matrix.T])


    def get_chi_squared_scores(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        observed = self.get_observed_quantities(pfi)
        expected = self.get_expected_quantities(pfi)

        def is_empty_feature(feature: Feature):
            return feature.variable_mask.count() == 0

        if any(is_empty_feature(f) for f in pfi.features):
            print("Empty feature detected")

        o_minus_e = observed-expected
        return (o_minus_e)/np.sqrt(expected)



    def __init__(self):
        self.cached_marginals = PPICachedData(self.get_marginal_probabilities)

    def __repr__(self):
        return f"Artefact"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return self.get_chi_squared_scores(pfi)

    def describe_score(self, given_score) -> str:
        return f"artefact= {given_score:.2f}"



class InteractionSimplified(MeasurableCriterion):
    """this just uses the average fitness"""
    cached_pX1s: PPICachedData
    cached_normalised_fitnesses: PPICachedData

    def get_pX1s(self, ppi: PrecomputedPopulationInformation) -> FlatArray:
        normalised_fitnesses = self.cached_normalised_fitnesses.get_data_for_ppi(ppi)

        def get_P1X_for_column(val_column: int) -> float:
            where_val_is_used = ppi.candidate_matrix[:, val_column] == 1
            return np.mean(normalised_fitnesses, where=where_val_is_used)

        amount_of_values = ppi.search_space.total_cardinality
        return np.array([get_P1X_for_column(val_column) for val_column in range(amount_of_values)])

    def get_p1Xs_for_feature_adjusted_by_fitness(self, feature_column_index: int,
                                                 pfi: PrecomputedFeatureInformation,
                                                 normalised_fitnesses: FlatArray) -> (float, np.ndarray):
        errors_for_feature = pfi.feature_presence_error_matrix[:, feature_column_index]
        less_than_two_errors = errors_for_feature < 2
        feature = pfi.feature_matrix[:, feature_column_index]
        variable_is_used = feature == 1

        rows_with_less_than_two_errors = pfi.candidate_matrix[less_than_two_errors][:, variable_is_used]
        fitnesses_with_less_than_two_errors = normalised_fitnesses[less_than_two_errors]
        error_counts = errors_for_feature[less_than_two_errors]

        with_one_error = error_counts == 1
        candidates_with_one_error = rows_with_less_than_two_errors[with_one_error]
        fitnesses_with_one_error = fitnesses_with_less_than_two_errors[with_one_error]

        fitnesses_with_zero_errors = fitnesses_with_less_than_two_errors[error_counts == 0]

        individual_errors = (1 - candidates_with_one_error) * feature[variable_is_used]
        fitnesses_where_no_errors = individual_errors * fitnesses_with_one_error.reshape((-1, 1))

        p11 = np.mean(fitnesses_with_zero_errors)

        samples_for_each_column = np.sum(individual_errors, axis=0)
        samples_for_each_var = samples_for_each_column + len(fitnesses_with_zero_errors)
        p1Xs = (np.sum(fitnesses_where_no_errors, axis=0) + np.sum(p11)) / samples_for_each_var

        return p11, p1Xs

    def get_remapped_fitnesses(self, ppi: PrecomputedPopulationInformation) -> np.ndarray:
        return utils.remap_array_in_zero_one(ppi.fitness_array)

    def __init__(self):
        self.cached_normalised_fitnesses = PPICachedData(self.get_remapped_fitnesses)
        self.cached_pX1s = PPICachedData(self.get_pX1s)


    def weakest_link_for_feature(self, feature_column_index: int,
                                 pfi: PrecomputedFeatureInformation) -> float:
        feature: HotEncodedFeature = pfi.feature_matrix[:, feature_column_index]
        if not any(feature):
            return 0
        all_pX1s: FlatArray = self.cached_pX1s.get_data_for_ppi(pfi.precomputed_population_information)
        normalised_fitnesses: FlatArray = self.cached_normalised_fitnesses.get_data_for_ppi(
            pfi.precomputed_population_information)

        pX1s = all_pX1s[feature == 1]
        p11, p1Xs = self.get_p1Xs_for_feature_adjusted_by_fitness(feature_column_index, pfi, normalised_fitnesses)

        if p11 == 0:
            return 0
        max_pX1_p1X = np.max(pX1s * p1Xs)
        result = p11 * np.log2(p11 / max_pX1_p1X)

        return result

    def __repr__(self):
        return f"WeakestLink(P)"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        scores = np.array([self.weakest_link_for_feature(feature_index, pfi)
                           for feature_index in range(pfi.amount_of_features)])

        return scores

    def describe_score(self, given_score) -> str:
        return f"Weakest_Link(P) = {given_score:.2f}"


class MutualInformation(MeasurableCriterion):
    cached_pXBs: PPICachedData

    def get_p_XBs(self, ppi: PrecomputedPopulationInformation) -> FlatArray:
        """ In this case we're returning the marginal probabilities"""
        return np.sum(ppi.candidate_matrix, axis=0) / ppi.sample_size

    def get_p_AB_AXs_XBs_for_feature(self, feature_column_index: int,
                                     pfi: PrecomputedFeatureInformation,
                                     cached_p_XBs: FlatArray) -> (float, FlatArray, FlatArray):
        # first, we only consider rows where there are less than 2 errors
        errors_for_feature = pfi.feature_presence_error_matrix[:, feature_column_index]
        less_than_two_errors = errors_for_feature < 2
        feature = pfi.feature_matrix[:, feature_column_index]
        which_variables_are_fixed = feature == 1  # used to select columns with fixed variables
        rows_with_less_than_two_errors = pfi.candidate_matrix[less_than_two_errors][:, which_variables_are_fixed]
        error_counts_where_less_than_two = errors_for_feature[less_than_two_errors]

        # we keep all the rows with exactly one error
        rows_with_one_error = rows_with_less_than_two_errors[error_counts_where_less_than_two == 1]
        where_single_errors_are = (1 - rows_with_one_error) * feature[which_variables_are_fixed]


        # calculate the counts
        c_AB = len(rows_with_less_than_two_errors) - len(rows_with_one_error)  # amount of rows that are exact matches
        count_where_one_error_elsewhere = np.sum(where_single_errors_are, axis=0)
        c_AX = c_AB + count_where_one_error_elsewhere

        # calculate the probabilities
        p_AB = c_AB / pfi.sample_size
        p_AXs = c_AX / pfi.sample_size

        p_XBs = cached_p_XBs[which_variables_are_fixed]

        return p_AB, p_AXs, p_XBs

    def get_remapped_fitnesses(self, ppi: PrecomputedPopulationInformation) -> np.ndarray:
        return utils.remap_array_in_zero_one(ppi.fitness_array)

    def __init__(self):
        self.cached_pXBs = PPICachedData(self.get_p_XBs)


    def get_lowest_information_share_for_feature(self, feature_column_index: int,
                                                 pfi: PrecomputedFeatureInformation,
                                                 cached_p_XBs: FlatArray) -> float:
        p_AB, p_AXs, p_XBs = self.get_p_AB_AXs_XBs_for_feature(feature_column_index, pfi, cached_p_XBs)
        if len(p_XBs) == 0:
            return 0

        max_p_AX_p_BX = np.max(p_AXs * p_XBs)
        result = p_AB * np.log2(p_AB / max_p_AX_p_BX)

        return result

    def __repr__(self):
        return f"MutualInformation"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        cached_p_XBs: FlatArray = self.cached_pXBs.get_data_for_ppi(pfi.precomputed_population_information)
        scores = np.array([self.get_lowest_information_share_for_feature(feature_index, pfi, cached_p_XBs)
                           for feature_index in range(pfi.amount_of_features)])

        return scores

    def describe_score(self, given_score) -> str:
        return f"MutualInformation = {given_score:.2f}"