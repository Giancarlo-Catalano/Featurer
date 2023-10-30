import random
from copy import copy
from typing import Iterable

import numpy as np
from bitarray import bitarray, frozenbitarray

import utils
from SearchSpace import SearchSpace, Candidate
from Version_E import HotEncoding
from Version_E.Feature import Feature
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation


class SHAPValue(MeasurableCriterion):
    sample_size: int

    def __init__(self, sample_size=360):
        self.sample_size = sample_size

    def __repr__(self):
        return f"SHAPValue(sample_size = {self.sample_size}"

    def bitarray_with_only_certain_vars_set(self, vars_to_set: list[int], total_size: int) -> frozenbitarray:
        result = bitarray(total_size)
        result.setall(0)
        for var in vars_to_set:
            result[var] = 1
        return frozenbitarray(result)

    def get_sampled_features_from_feature(self, feature: Feature,
                                          ppi: PrecomputedPopulationInformation) -> list[Feature]:
        # first we choose the candidates to apply our tests on
        indexes_to_keep = [random.randrange(ppi.sample_size) for _ in range(self.sample_size)]
        selected_candidate_matrix = copy(ppi.candidate_matrix[indexes_to_keep, :])

        # then we remove some variables, and get the features
        vars_to_remove = [var_index for var_index, is_used in enumerate(feature.variable_mask) if is_used]
        columns_to_remove = [ppi.search_space.precomputed_offsets[var] + val
                             for var in vars_to_remove
                             for val in range(ppi.search_space.cardinalities[var])]
        selected_candidate_matrix[:, columns_to_remove] = 0  # this is redundant but it feels necessary


        def feature_from_row(row) -> Feature:
            return HotEncoding.feature_from_hot_encoding(row, ppi.search_space)

        return [feature_from_row(row) for row in selected_candidate_matrix]


    def get_raw_score_for_feature(self, feature: Feature, ppi: PrecomputedPopulationInformation) -> float:
        sampled_features = self.get_sampled_features_from_feature(feature, ppi)
        sampled_features_pfi = PrecomputedFeatureInformation(ppi, sampled_features)
        standard_deviations = sampled_features_pfi.sd_for_each_feature
        return np.mean(standard_deviations)

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        ppi = pfi.precomputed_population_information
        return np.array([self.get_raw_score_for_feature(feature, ppi) for feature in pfi.features])

    def describe_score(self, given_score) -> str:
        return f"Has (estimated SHAP value of  {given_score})"
