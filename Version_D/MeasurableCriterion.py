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


def compute_observed_distribution_marginal_probabilities(pfi: PrecomputedFeatureInformation) -> tuple[tuple[float]]:
    sum_in_hot_encoding_order: np.ndarray[float] = np.sum(pfi.candidate_matrix, axis=0)
    search_space = pfi.search_space

    def observed_proportions_for_variable(var_index) -> tuple[float]:
        start, end = search_space.precomputed_offsets[var_index: var_index + 2]
        return tuple(sum_in_hot_encoding_order[start:end] / search_space.cardinalities[var_index])

    return tuple(observed_proportions_for_variable(var_index) for var_index in range(search_space.dimensions))


def compute_uniform_distribution_marginal_probabilities(search_space: SearchSpace) -> tuple[tuple[float]]:
    def compute_uniform_distribution_of_variable(cardinality) -> tuple[float]:
        probability = 1 / cardinality
        return tuple([probability] * cardinality)

    return tuple(compute_uniform_distribution_of_variable(cardinality)
                 for cardinality in search_space.cardinalities)


def compute_expected_probabilities(pfi: PrecomputedFeatureInformation, marginals: tuple[tuple[float]]) -> np.ndarray:
    # TODO this needs to be tested
    marginal_array = np.array(utils.concat_tuples(marginals))  # I'm aware that this is silly
    exponents_of_marginals = np.log2(marginal_array)
    sum_of_exponents = utils.weighted_sum_of_columns(exponents_of_marginals, pfi.feature_matrix.T)
    expected_probabilities = np.power(2, sum_of_exponents)
    return expected_probabilities


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
        marginal_probabilities: tuple[tuple[float]] = tuple()  # dummy value
        if self.relative_to_uniform:
            marginal_probabilities = compute_uniform_distribution_marginal_probabilities(pfi.search_space)
        else:
            marginal_probabilities = compute_observed_distribution_marginal_probabilities(pfi)

        expected_probabilities = compute_expected_probabilities(pfi, marginal_probabilities)
        expected_counts = expected_probabilities * pfi.sample_size
        observed_counts = pfi.count_for_each_feature

        signed_chi_squareds = utils.signed_chi_squared(observed_counts, expected_counts)
        return signed_chi_squareds
