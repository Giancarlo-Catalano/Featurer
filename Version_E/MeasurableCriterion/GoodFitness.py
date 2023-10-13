import numpy as np
import utils
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation


class HighFitness(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "Mean Fitness"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return pfi.mean_fitness_for_each_feature


    def describe_score(self, given_score) -> str:
        return f"Average fitness = {given_score:.2f}"


def compute_t_scores(pfi: PrecomputedFeatureInformation, signed=False) -> np.ndarray:
    sd_over_root_n = utils.divide_arrays_safely(pfi.sd_for_each_feature, np.sqrt(pfi.count_for_each_feature))
    numerator = pfi.mean_fitness_for_each_feature - pfi.population_mean
    if not signed:
        numerator = np.abs(numerator)
    t_scores = utils.divide_arrays_safely(numerator, sd_over_root_n)
    return t_scores


class ConsistentFitness(MeasurableCriterion):
    signed: bool
    def __init__(self, signed=False):
        self.signed = signed

    def __repr__(self):
        return "Fitness Consistency"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return compute_t_scores(pfi, self.signed)

    def describe_score(self, given_score) -> str:
        return f"Consistent with t-value = {given_score:.2f}"


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
        return f"Fitness higher than average {given_score*100:.2f}% of the time"
