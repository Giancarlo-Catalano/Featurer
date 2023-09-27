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


def compute_t_scores(pfi: PrecomputedFeatureInformation) -> np.ndarray:
    sd_over_root_n = utils.divide_arrays_safely(pfi.sd_for_each_feature, np.sqrt(pfi.count_for_each_feature))
    t_scores = utils.divide_arrays_safely(np.abs(pfi.mean_fitness_for_each_feature - pfi.population_mean), sd_over_root_n)
    return t_scores


class ConsistentFitness(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "Fitness Consistency"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return compute_t_scores(pfi)

    def describe_score(self, given_score) -> str:
        return f"Consistent with t-value = {given_score:.2f}"
