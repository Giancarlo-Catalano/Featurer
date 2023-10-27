import numpy as np

import utils
from Version_E.Feature import Feature
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation


def remap_array_in_zero_one(input_array: np.ndarray):
    # TODO: get rid of the original function once it's not used elsewhere
    return utils.remap_array_in_zero_one(input_array)


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

    def describe_score(self, given_score) -> str:
        raise Exception("An implementation of MeasurableCriterion does not implement describe_score")

    def get_single_raw_score(self, feature: Feature, ppi: PrecomputedPopulationInformation) -> float:
        pfi = PrecomputedFeatureInformation(ppi, [feature])
        return self.get_raw_score_array(pfi)[0]

    def describe_feature(self, feature: Feature, ppi: PrecomputedPopulationInformation) -> str:
        raw_score = self.get_single_raw_score(feature, ppi)
        return self.describe_score(raw_score)