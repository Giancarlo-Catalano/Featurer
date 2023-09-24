import numpy as np
from typing import Iterable, Any

import SearchSpace
import utils
from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from SearchSpace import SearchSpace

class ProportionCriterion(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "Proportion"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return pfi.count_for_each_feature

    def describe_score(self, given_score) -> str:
        return f"Appears {given_score} times in the population"

def compute_uniform_distribution_marginal_probabilities(search_space: SearchSpace) -> np.ndarray:
    return np.ndarray([cardinality] * cardinality for cardinality in search_space.cardinalities)


def compute_expected_probabilities(pfi: PrecomputedFeatureInformation, marginal_array: np.ndarray) -> np.ndarray:
    # TODO this needs to be tested thoroughly!
    exponents_of_marginals = np.log2(marginal_array)
    sum_of_exponents = utils.weighted_sum_of_columns(exponents_of_marginals, pfi.feature_matrix.T)
    expected_probabilities = np.power(2, sum_of_exponents)

    which_to_multiply = np.array(pfi.feature_matrix, dtype=bool)
    marginals_in_every_column = np.tile(marginal_array, (pfi.amount_of_features, 1)).T
    product_of_each_row = np.product(marginals_in_every_column, where=which_to_multiply, axis=0)
    return product_of_each_row


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


    def describe_score(self, given_score) -> str:
        return f"Representation has chi-score = {given_score}"



