import Feature
import numpy as np
from typing import Iterable, Any

import utils


def remap_array_in_zero_one(input: np.ndarray):
    # TODO: get rid of the original function once it's not used elsewhere
    return utils.remap_array_in_zero_one(input)


class MeasurableCriterion:
    """ A criterion which makes a feature meaningful"""
    def __repr__(self):
        """ Return a string which describes the Criterion, eg 'Robustness' """
        raise Exception("Error: a realisation of MeasurableCriterion does not implement __repr__")

    def get_raw_score_array(self, precomputed_feature_information_in_population) -> np.ndarray:
        """ Return a numpy array which contains a numerical score for each feature in the input.
            NOTE: the score should INCREASE as the criteria is being satisfied
        """
        raise Exception("Error: a realisation of MeasurableCriterion does not implement get_score_array")


    def get_score_array(self, precomputed_feature_information_in_population) -> np.ndarray:
        """ Returns the scores which correlate with the criterion
            And they will all be in the range [0, 1]"""
        raw_scores = self.get_raw_score_array(precomputed_feature_information_in_population)
        return remap_array_in_zero_one(raw_scores)

    def get_inverse_score_array(self, precomputed_feature_information_in_population) -> np.ndarray:
        """ Returns the scores which correlate NEGATIVELY with the criterion
                    And they will all be in the range [0, 1]"""
        return 1.0 - self.get_score_array(precomputed_feature_information_in_population)



class ExplainabilityCriterion(MeasurableCriterion):
    complexity_function: Any
    def __init__(self, complexity_function: Any):
        self.complexity_function = complexity_function

    def __repr__(self):
        return "Explainability"




