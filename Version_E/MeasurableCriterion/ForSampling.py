import numpy as np

from Version_E.Feature import Feature
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation


class Completeness(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "Completeness"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return np.array([feature.variable_mask.count() for feature in pfi.features])

    def describe_score(self, given_score) -> str:
        return f"Has {given_score} variables set"


class ExpectedFitness(MeasurableCriterion):
    reference_features: list[Feature]
    criterion: MeasurableCriterion

    reference_feature_scores: list[float]

    pseudo_ppi: PrecomputedPopulationInformation

    def __init__(self, pfi: PrecomputedFeatureInformation, criterion: MeasurableCriterion):
        self.reference_features = pfi.features
        self.criterion = criterion
        self.reference_feature_scores = list(criterion.get_score_array(pfi))

        self.pseudo_ppi = PrecomputedPopulationInformation(search_space=pfi.search_space,
                                                           population_sample=self.reference_features,
                                                           fitness_list=self.reference_feature_scores,
                                                           population_is_full_solutions=False)

    def __repr__(self):
        return "ExpectedFitness"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        pseudo_pfi = PrecomputedFeatureInformation(self.pseudo_ppi, pfi.features)
        return pseudo_pfi.mean_fitness_for_each_feature
