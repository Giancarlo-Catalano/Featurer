import numpy as np

from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from SearchSpace import SearchSpace
from Version_E.Feature import Feature
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from typing import Any


class Explainability(MeasurableCriterion):
    complexity_function: Any

    def __init__(self, problem: CombinatorialProblem):
        self.complexity_function = problem.get_complexity_of_feature

    def __repr__(self):
        return "Explainability"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return np.array([-self.complexity_function(feature.to_legacy_feature()) for feature in pfi.features])

    def describe_score(self, given_score) -> str:
        return f"Complexity = {-given_score:.2f}"


class TrivialComplexity(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "(Trivial) Complexity"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return np.ndarray([feature.variable_mask.count() for feature in pfi.features])

    def describe_score(self, given_score) -> str:
        return f"Trivial complexity = {given_score}"


class TrivialExplainability(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "(Trivial) Explainability"

    def explainability_of_feature(self, feature: Feature, total_amount_of_vars: int) -> float:
        amount_of_vars = feature.variable_mask.count()
        if amount_of_vars < 2:
            return amount_of_vars
        else:
            return total_amount_of_vars - amount_of_vars

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        amount_of_vars = pfi.search_space.dimensions
        return np.array([self.explainability_of_feature(feature, amount_of_vars)
                         for feature in pfi.features])


    def get_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        amount_of_vars = np.sum(pfi.feature_matrix, axis=0)
        return 1 - (amount_of_vars / pfi.search_space.dimensions)

    def describe_score(self, given_score) -> str:
        return f"Trivial explainability = {given_score:.2f}"



class TargetSize(MeasurableCriterion):
    target_size: int

    def __init__(self, target_size: int):
        self.target_size = target_size

    def __repr__(self):
        return f"TargetSize({self.target_size})"


    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        amount_of_values = np.sum(pfi.feature_matrix, axis=0)
        return -np.abs(amount_of_values-self.target_size)

    def describe_score(self, given_score) -> str:
        return f"Distance from target size = {-given_score}"