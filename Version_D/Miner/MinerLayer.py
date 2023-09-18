import numpy as np

import utils
from SearchSpace import SearchSpace
from Version_D import MeasurableCriterion
from Version_D.Feature import Feature

from Version_D.Miner import LayerMixer
from Version_D.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_D.PrecomputedPopulationInformation import PrecomputedPopulationInformation


class MinerLayer:
    """this is a data structure to store a list of features and their scores"""
    """ The scores indicate how good a feature is, in terms of explainability and either fitness or novelty,
        These scores go from 0 to 1 and are also used to sample the features using weights.
        The main purpose of this class is to select a random feature and use it as a parent somewhere else
    """
    features: list[Feature]
    scores: np.array

    precomputed_cumulative_list: np.array

    def __init__(self, features, scores):
        self.features = features
        self.scores = scores
        self.precomputed_cumulative_list = np.cumsum(scores)


