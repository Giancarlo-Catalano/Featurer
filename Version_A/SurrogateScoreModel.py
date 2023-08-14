import math

import CooccurrenceModel
import ProgressiveFeatures
import numpy as np
import HotEncoding
import utils


class SurrogateScoreModel:
    cooccurrence_model: CooccurrenceModel.CooccurrenceModel
    hot_encoder: HotEncoding.HotEncoder

    def __init__(self, feature_detection_model: ProgressiveFeatures.ProgressiveFeatures):
        scores_of_features = feature_detection_model.get_scores_of_features_in_model(
            adjust_scores_considering_complexity=False)
        features = feature_detection_model.cooccurrence_model.feature_group.hot_features  # in future versions, a conversion from a set to a list will be necessary
        proportion_to_keep = 1
        amount_to_keep = math.ceil(proportion_to_keep * len(scores_of_features))
        features_with_scores = list(zip(features, scores_of_features))
        features_with_scores.sort(key=utils.second, reverse=True)
        features_with_scores = features_with_scores[:amount_to_keep]

        (just_features, weights) = utils.unzip(features_with_scores)

        self.cooccurrence_model = CooccurrenceModel.CooccurrenceModel(feature_detection_model.search_space,
                                                    just_features,
                                                    feature_detection_model.candidate_matrix,
                                                    feature_detection_model.fitness_list)

        self.cooccurrence_model.cooccurrence_matrix = np.diag(weights) @ self.cooccurrence_model.cooccurrence_matrix @ np.diag(weights)
        self.hot_encoder = HotEncoding.HotEncoder(feature_detection_model.search_space)

    def get_surrogate_score(self, combinatorial_candidate):
        return self.cooccurrence_model.score_of_raw_candidate_vector(self.hot_encoder.candidate_to_hot_encoding(combinatorial_candidate))
