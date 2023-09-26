import numpy as np

from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from typing import Any


class Explainability(MeasurableCriterion):
    complexity_function: Any

    def __init__(self, problem: CombinatorialProblem):
        self.complexity_function = problem.get_complexity_of_feature

    def __repr__(self):
        return "Explainability.py"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return np.array([-self.complexity_function(feature.to_legacy_feature()) for feature in pfi.features])

    def describe_score(self, given_score) -> str:
        return f"Complexity = {-given_score}"