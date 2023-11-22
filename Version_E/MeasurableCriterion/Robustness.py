import numpy as np

import utils
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.MeasurableCriterion.CriterionUtilities import Not


def get_fuzzy_match_matrix(pfi: PrecomputedFeatureInformation, min_errors: int, max_errors: int):
    at_least_min_errors = pfi.feature_presence_error_matrix >= min_errors
    at_most_max_errors = pfi.feature_presence_error_matrix <= max_errors
    match_matrix = np.array(np.logical_and(at_least_min_errors, at_most_max_errors), dtype=float)
    return match_matrix


def get_mean_of_fuzzy_match_matrix(fuzzy_matrix: np.ndarray, pfi: PrecomputedFeatureInformation) -> np.ndarray:
    sum_of_fitnesses = utils.weighted_sum_of_rows(fuzzy_matrix,
                                                  pfi.fitness_array)

    count_for_each_feature = np.sum(fuzzy_matrix, axis=0)

    return utils.divide_arrays_safely(sum_of_fitnesses, count_for_each_feature)


def get_sd_of_fuzzy_match_matrix(fuzzy_matrix: np.ndarray, pfi: PrecomputedFeatureInformation) -> np.ndarray:
    sum_of_fitnesses = utils.weighted_sum_of_rows(fuzzy_matrix,
                                                  pfi.fitness_array)
    count_for_each_feature = np.sum(fuzzy_matrix, axis=0)
    mean_fitness_for_each_feature = utils.divide_arrays_safely(sum_of_fitnesses, count_for_each_feature)

    fitness_of_candidate_minus_feature_mean = np.subtract.outer(pfi.fitness_array,
                                                                mean_fitness_for_each_feature)

    fitness_of_candidate_minus_feature_mean *= fuzzy_matrix
    fitness_of_candidate_minus_feature_mean = np.square(
        fitness_of_candidate_minus_feature_mean)  # may God forgive me...

    numerators = np.sum(fitness_of_candidate_minus_feature_mean, axis=0)

    return np.sqrt(utils.divide_arrays_safely(numerators, count_for_each_feature - 1))  # should this count-1


class Robustness(MeasurableCriterion):
    min_amount_of_differences: int
    max_amount_of_differences: int

    def __init__(self, min_amount_of_differences=1, max_amount_of_differences=1):
        self.min_amount_of_differences = min_amount_of_differences
        self.max_amount_of_differences = max_amount_of_differences

    def __repr__(self):
        return f"Robustness (errors in [{self.min_amount_of_differences}, {self.max_amount_of_differences}])"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        normal_means = pfi.mean_fitness_for_each_feature
        fuzzy_match_matrix = get_fuzzy_match_matrix(pfi, self.min_amount_of_differences, self.max_amount_of_differences)
        fuzzy_mean = get_mean_of_fuzzy_match_matrix(fuzzy_match_matrix, pfi)

        # the denominator there is to normalise the "deviation": without it, smaller average fitnesses deviate relatively more
        return (normal_means - fuzzy_mean)# / (1 + np.abs(normal_means) + np.abs(fuzzy_mean))

    def get_raw_score_array_experimental(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        fuzzy_match_matrix = get_fuzzy_match_matrix(pfi, self.min_amount_of_differences, self.max_amount_of_differences)
        sds = get_sd_of_fuzzy_match_matrix(fuzzy_match_matrix, pfi)
        return -sds

    def describe_score(self, given_score) -> str:
        return f"Robustness score [{self.min_amount_of_differences}, {self.max_amount_of_differences}] = {given_score:.2f}"

def Atomicity():
    return Robustness(1, 1)
