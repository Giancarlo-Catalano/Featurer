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


class FitnessConsistencyCriterion(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "Fitness Consistency"

    def compute_t_scores(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        sd_over_root_n = utils.divide_arrays_safely(pfi.sd_for_each_feature, np.sqrt(pfi.count_for_each_feature))
        t_scores = utils.divide_arrays_safely(pfi.mean_fitness_for_each_feature - pfi.population_mean, sd_over_root_n)
        return t_scores

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return self.compute_t_scores(pfi)

class PopularityCriterion(MeasurableCriterion):
    marginal_probabilities: tuple[tuple[float]] # marginal probability for each (var, val)


    def compute_uniform_distribution_marginal_probabilities(self, search_space: SearchSpace) -> tuple[tuple[float]]:
        def compute_uniform_distribution_of_variable(cardinality) -> tuple[float]:
            probability = 1 / cardinality
            return tuple([probability] * cardinality)

        return tuple(compute_uniform_distribution_of_variable(cardinality)
                     for cardinality in search_space.cardinalities)


    def compute_observed_distribution_marginal_probabilities(self, search_space: SearchSpace, pfi: PrecomputedFeatureInformation) -> tuple[tuple[float]]:
        sum_in_hot_encoding_order: list[float] = np.sum(self.candidate_matrix, axis=0).tolist()

        def counts_for_each_variable(var_index):
            start, end = self.search_space.precomputed_offsets[var_index: var_index + 2]
            return sum_in_hot_encoding_order[start:end]

        return [counts_for_each_variable(var_index) for var_index in range(self.search_space.dimensions)]

    def __init__(self, relative_to_uniform_distribution = True,
                       relative_to_observed_marginals = False):
