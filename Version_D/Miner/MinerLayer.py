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

    @classmethod
    def make_0_parameter_layer(cls, search_space: SearchSpace):
        empty_feature = Feature.empty_feature(search_space)
        scores = np.array(1)  # dummy value
        return cls([empty_feature], scores)

    @classmethod
    def make_by_mixing(cls, mother_layer, father_layer,
                       ppi: PrecomputedPopulationInformation,
                       criteria_and_weights: MeasurableCriterion.LayerScoringCriteria,
                       parent_pair_iterator: LayerMixer.ParentPairIterator,
                       how_many_to_generate: int,
                       how_many_to_keep: int):
        # breed
        offspring = LayerMixer.get_layer_offspring(mother_layer, father_layer,
                                                   parent_pair_iterator, requested_amount=how_many_to_generate)
        # assess
        pfi: PrecomputedFeatureInformation = PrecomputedFeatureInformation(ppi, offspring)
        scores = MeasurableCriterion.compute_scores_for_features(pfi, criteria_and_weights)

        # select
        sorted_by_with_score = sorted(zip(offspring, scores), key=utils.second, reverse=True)
        features, scores_list = utils.unzip(sorted_by_with_score[:how_many_to_keep])
        return cls(features, np.array(scores_list))
