import Feature
import numpy as np
from typing import Iterable, Any

import SearchSpace
import utils
from Version_D.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from SearchSpace import SearchSpace


def remap_array_in_zero_one(input: np.ndarray):
    # TODO: get rid of the original function once it's not used elsewhere
    return utils.remap_array_in_zero_one(input)


class MeasurableCriterion:
    """ A criterion which makes a feature meaningful"""

    def __repr__(self):
        """ Return a string which describes the Criterion, eg 'Robustness' """
        raise Exception("Error: a realisation of MeasurableCriterion does not implement __repr__")

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        """ Return a numpy array which contains a numerical score for each feature in the input.
            NOTE: the score should INCREASE as the criteria is being satisfied
        """
        raise Exception("Error: a realisation of MeasurableCriterion does not implement get_score_array")

    def get_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        """ Returns the scores which correlate with the criterion
            And they will all be in the range [0, 1]"""
        raw_scores = self.get_raw_score_array(pfi)
        return remap_array_in_zero_one(raw_scores)

    def get_inverse_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        """ Returns the scores which correlate NEGATIVELY with the criterion
                    And they will all be in the range [0, 1]"""
        return 1.0 - self.get_score_array(pfi)


class ExplainabilityCriterion(MeasurableCriterion):
    complexity_function: Any

    def __init__(self, complexity_function: Any):
        self.complexity_function = complexity_function

    def __repr__(self):
        return "Explainability"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return np.array([self.complexity_function(feature) for feature in pfi.features])


class MeanFitnessCriterion(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "Mean Fitness"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return pfi.mean_fitness_for_each_feature


def compute_t_scores(pfi: PrecomputedFeatureInformation) -> np.ndarray:
    sd_over_root_n = utils.divide_arrays_safely(pfi.sd_for_each_feature, np.sqrt(pfi.count_for_each_feature))
    t_scores = utils.divide_arrays_safely(pfi.mean_fitness_for_each_feature - pfi.population_mean, sd_over_root_n)
    return t_scores


class FitnessConsistencyCriterion(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "Fitness Consistency"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return compute_t_scores(pfi)


def compute_uniform_distribution_marginal_probabilities(search_space: SearchSpace) -> np.ndarray:
    return np.ndarray([cardinality] * cardinality for cardinality in search_space.cardinalities)


def compute_expected_probabilities(pfi: PrecomputedFeatureInformation, marginal_array: np.ndarray) -> np.ndarray:
    # TODO this needs to be tested
    exponents_of_marginals = np.log2(marginal_array)
    sum_of_exponents = utils.weighted_sum_of_columns(exponents_of_marginals, pfi.feature_matrix.T)
    expected_probabilities = np.power(2, sum_of_exponents)
    return expected_probabilities


def signed_chi_squared(observed, expected):
    def signed_squared(x):
        return x * np.abs(x)

    return (signed_squared(observed - expected)) / expected


class PopularityCriterion(MeasurableCriterion):
    relative_to_uniform: bool

    def __init__(self, relative_to_uniform_distribution=True):
        self.relative_to_uniform = relative_to_uniform_distribution

    def __repr__(self):
        if self.relative_to_uniform:
            return "Popularity (relative to uniform distribution)"
        else:
            return "Popularity (relative to marginal distribution)"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        if self.relative_to_uniform:
            marginal_probabilities = compute_uniform_distribution_marginal_probabilities(pfi.search_space)
        else:
            marginal_probabilities = pfi.precomputed_marginal_probabilities

        expected_probabilities = compute_expected_probabilities(pfi, marginal_probabilities)
        expected_counts = expected_probabilities * pfi.sample_size
        observed_counts = pfi.count_for_each_feature

        signed_chi_squareds = signed_chi_squared(observed_counts, expected_counts)
        return signed_chi_squareds


class ProportionCriterion(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "Proportion"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return pfi.count_for_each_feature


def compute_phi_scores(pfi: PrecomputedFeatureInformation):
    count_for_each_value = np.sum(pfi.candidate_matrix, axis=0)
    absence_count_for_each_value = pfi.sample_size - count_for_each_value

    products_of_counts = np.power(2, utils.weighted_sum_of_columns(np.log2(count_for_each_value),
                                                                   pfi.feature_matrix.T))
    products_of_absences = np.power(2, utils.weighted_sum_of_columns(np.log2(absence_count_for_each_value),
                                                                     pfi.feature_matrix.T))

    n = pfi.sample_size
    n_all = pfi.count_for_each_feature

    numerators = (n * n_all - products_of_counts)
    denominators = np.sqrt(products_of_counts * products_of_absences)

    return utils.divide_arrays_safely(numerators, denominators, 0)


class CorrelationCriterion(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr(self):
        return "Correlation"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return compute_phi_scores(pfi)


def get_fuzzy_match_matrix(pfi: PrecomputedFeatureInformation, min_errors: int, max_errors: int):
    at_least_min_errors = pfi.feature_presence_error_matrix >= min_errors
    at_most_min_errors = pfi.feature_presence_error_matrix <= max_errors
    match_matrix = np.array(np.logical_and(at_most_min_errors, at_most_min_errors), dtype=float)
    return match_matrix


def get_mean_of_fuzzy_match_matrix(fuzzy_matrix: np.ndarray, pfi: PrecomputedFeatureInformation):
    sum_of_fitnesses = utils.weighted_sum_of_rows(fuzzy_matrix,
                                                  pfi.fitness_array)

    count_for_each_feature = np.sum(fuzzy_matrix, axis=0)

    return utils.divide_arrays_safely(sum_of_fitnesses, count_for_each_feature)


class RobustnessCriterion(MeasurableCriterion):
    min_amount_of_differences: int
    max_amount_of_differences: int

    def __init__(self, min_amount_of_differences = 1, max_amount_of_differences = 1):
        self.min_amount_of_differences = min_amount_of_differences
        self.max_amount_of_differences = max_amount_of_differences

    def __repr__(self):
        return f"Robustness (errors in [{self.min_amount_of_differences}, {self.max_amount_of_differences}])"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        normal_means = pfi.mean_fitness_for_each_feature
        fuzzy_match_matrix = get_fuzzy_match_matrix(pfi, self.min_amount_of_differences, self.max_amount_of_differences)
        fuzzy_mean = get_mean_of_fuzzy_match_matrix(fuzzy_match_matrix, pfi)

        return (normal_means - fuzzy_mean) / (1 + np.abs(normal_means) + np.abs(fuzzy_mean))