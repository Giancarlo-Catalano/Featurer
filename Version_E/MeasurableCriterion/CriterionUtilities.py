from typing import Callable, Optional

import numpy as np

import utils
from Version_E.Feature import Feature
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation


class Not(MeasurableCriterion):
    criterion: MeasurableCriterion

    def __init__(self, criterion: MeasurableCriterion):
        self.criterion = criterion

    def __repr__(self):
        return "Not"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return - self.criterion.get_raw_score_array(pfi)

    def describe_feature(self, feature: Feature, ppi: PrecomputedPopulationInformation) -> str:
        return self.criterion.describe_feature(feature, ppi)


class Balance(MeasurableCriterion):
    criteria: list[MeasurableCriterion]
    weights: np.ndarray

    def __init__(self, criteria: list[MeasurableCriterion], weights=None):
        if weights is None:
            weights = [1 for criterion in criteria]

        self.weights = np.array(weights, dtype=float)

        self.criteria = criteria


    def __repr__(self):
        return "Balance of [" + \
                ", ".join(f"{criterion}({weight})" for criterion, weight in zip(self.criteria, self.weights)) + \
                "]"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        scores = [criterion.get_score_array(pfi) for criterion in self.criteria]
        atomic_scores = np.array(scores)
        return utils.weighted_average_of_rows(atomic_scores, self.weights)

    def describe_feature(self, feature: Feature, ppi: PrecomputedPopulationInformation) -> str:
        return ", ".join(criterion.describe_feature(feature, ppi) for criterion in self.criteria)


class Any(MeasurableCriterion):
    criteria: list[MeasurableCriterion]

    def __init__(self, criteria: list[MeasurableCriterion]):
        self.criteria = criteria

    def __repr__(self):
        return "Any of "+", ".join(f"{criterion}" for criterion in self.criteria)

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        atomic_scores = np.array([criterion.get_score_array(pfi) for criterion in self.criteria])
        return np.max(atomic_scores, axis=0)

    def describe_feature(self, feature: Feature, ppi: PrecomputedPopulationInformation) -> str:
        return ", ".join(criterion.describe_feature(feature, ppi) for criterion in self.criteria)


class All(MeasurableCriterion):
    criteria: list[MeasurableCriterion]

    def __init__(self, criteria: list[MeasurableCriterion]):
        self.criteria = criteria

    def __repr__(self):
        return "All of " + ", ".join(f"{criterion}" for criterion in self.criteria)

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        atomic_scores = np.array([criterion.get_score_array(pfi) for criterion in self.criteria])
        return np.min(atomic_scores, axis=0)

    def describe_feature(self, feature: Feature, ppi: PrecomputedPopulationInformation) -> str:
        return ", ".join(criterion.describe_feature(feature, ppi) for criterion in self.criteria)


class PPICachedData:
    """ this class is used to generate and cache information associated with a PPI
        usually, this is constraint information"""
    generator_function: Callable
    ppi_with_cache: list[(PrecomputedPopulationInformation, Any)]

    def __init__(self, generator_function):
        self.generator_function = generator_function
        self.ppi_with_cache = []

    def add_new_entry(self, new_ppi: PrecomputedPopulationInformation, new_data: Any):
        self.ppi_with_cache.append((new_ppi, new_data))

    def unsafe_get_data_for_ppi(self, ppi: PrecomputedPopulationInformation) -> Optional[Any]:
        for stored_ppi, cached_data in self.ppi_with_cache:
            if stored_ppi is ppi:  # that's right, we check for equality by only looking at the reference!!
                return cached_data

        return None

    def get_data_for_ppi(self, ppi) -> Any:
        collected = self.unsafe_get_data_for_ppi(ppi)
        if collected is None:
            new_data = self.generator_function(ppi)
            self.add_new_entry(ppi, new_data)
            return new_data
        return collected