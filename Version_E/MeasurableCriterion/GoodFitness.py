import numpy as np
import scipy

import utils
from Version_E.MeasurableCriterion.CriterionUtilities import PPICachedData
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation


class HighFitness(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "Mean Fitness"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return pfi.mean_fitness_for_each_feature

    def get_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        means = self.get_raw_score_array(pfi)
        min_fitness = np.min(pfi.fitness_array)
        max_fitness = np.max(pfi.fitness_array)

        position_within_range = (means - min_fitness) / (max_fitness - min_fitness)

        def smoothstep(x):
            return 3 * (x ** 2) - 2 * (x ** 3)

        return position_within_range

    def describe_score(self, given_score) -> str:
        return f"Average fitness = {given_score:.2f}"


def compute_t_scores(pfi: PrecomputedFeatureInformation, signed=False) -> np.ndarray:
    sd_over_root_n = utils.divide_arrays_safely(pfi.sd_for_each_feature, np.sqrt(pfi.count_for_each_feature))
    numerator = pfi.mean_fitness_for_each_feature - pfi.population_mean
    if not signed:
        numerator = np.abs(numerator)
    t_scores = utils.divide_arrays_safely(numerator, sd_over_root_n)
    return t_scores


def compute_welch_test_p_value_single(feature_col_index: int, pfi: PrecomputedFeatureInformation) -> float:
    present_where = np.array(pfi.feature_presence_matrix[:, feature_col_index], dtype=bool)
    absent_where = np.logical_not(present_where)

    fitnesses_where = pfi.fitness_array[present_where]
    fitnesses_where_not = pfi.fitness_array[absent_where]

    test_result = scipy.stats.ttest_ind(fitnesses_where, fitnesses_where_not, equal_var=False).pvalue
    if np.isnan(test_result):
        return 1
    else:
        return test_result





def compute_p_values(pfi: PrecomputedFeatureInformation, signed=False) -> np.ndarray:
    return np.array([compute_welch_test_p_value_single(feature_index, pfi)
                     for feature_index in range(pfi.amount_of_features)])


class ConsistentFitness(MeasurableCriterion):
    signed: bool

    def __init__(self, signed=False):
        self.signed = signed

    def __repr__(self):
        return "Fitness Consistency"

    def get_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        p_values = compute_p_values(pfi, self.signed)

        def normalise(x: np.ndarray):
            return 1-x

        return normalise(p_values)

    def describe_score(self, given_score) -> str:
        return f"The quality of the p value is {given_score:.2f}"






class FitnessHigherThanAverage(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "Fitness Higher Than average"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        """this will simply return a count of how the proportion of solutions containing the feature which have a higher than average fitness"""
        which_are_better_than_average = np.array(pfi.fitness_array > pfi.population_mean, dtype=float)
        count_of_greater_than_average = utils.weighted_sum_of_rows(pfi.feature_presence_matrix,
                                                                   which_are_better_than_average)
        return utils.divide_arrays_safely(count_of_greater_than_average, pfi.count_for_each_feature, else_value=0.0)

    def describe_score(self, given_score) -> str:
        return f"Fitness higher than average {given_score * 100:.2f}% of the time"


class WorstCase(MeasurableCriterion):
    cached_minimums = PPICachedData
    cached_maximums = PPICachedData

    def get_minimum(self, ppi: PrecomputedFeatureInformation):
        return np.min(ppi.fitness_array)

    def get_maximum(self, ppi: PrecomputedFeatureInformation):
        return np.max(ppi.fitness_array)

    def __init__(self):
        self.cached_minimums = PPICachedData(self.get_minimum)
        self.cached_maximums = PPICachedData(self.get_maximum)

    def __repr__(self):
        return "WorstCase"

    def get_minimum_fitness_for_feature(self, feature_col_index: int, pfi: PrecomputedFeatureInformation) -> float:
        where = np.array(pfi.feature_presence_matrix[:, feature_col_index], dtype=bool)
        absolute_minimum = 1000
        result = np.min(pfi.fitness_array, where=where, initial=absolute_minimum)

        return result

    def normalise_minimum(self, list_of_minimums: np.ndarray, pfi: PrecomputedFeatureInformation):
        ppi = pfi.precomputed_population_information
        known_min: float = self.cached_minimums.get_data_for_ppi(ppi)
        known_max: float = self.cached_maximums.get_data_for_ppi(ppi)

        position_in_range = (list_of_minimums - known_min) / (known_max - known_min)
        return position_in_range ** (1 / 5)

    def get_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        observed_minimums = np.array(
            [self.get_minimum_fitness_for_feature(index, pfi) for index in range(pfi.amount_of_features)])
        return self.normalise_minimum(observed_minimums, pfi)

    def describe_score(self, given_score) -> str:
        return f"The Absolute observed minimum is {given_score}"
