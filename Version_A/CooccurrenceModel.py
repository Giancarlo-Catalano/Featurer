import FeatureGroup
import utils
import numpy as np


class CooccurrenceModel:
    feature_group: FeatureGroup

    def get_cooccurrence_matrix(self, candidate_matrix, fitness_list):
        candidate_feature_matrix = self.feature_group.get_feature_vectors_for_candidates(candidate_matrix)

        def cooccurence_for_single_candidate(candidate_feature_vector, candidate_fitness):
            """ This is the most important line in the project"""
            return (utils.to_column_vector(candidate_feature_vector) @ utils.array_to_row_vector(
                candidate_feature_vector)) * candidate_fitness

        cooccurence_matrix = utils.zero_matrix(self.feature_group.amount_of_features)
        for single_cand_matrix, fitness in zip(candidate_feature_matrix, fitness_list):
            cooccurence_matrix += cooccurence_for_single_candidate(single_cand_matrix, fitness)
        return cooccurence_matrix

    def __init__(self, search_space, hot_encoded_features, candidate_matrix, fitness_list):
        self.feature_group = FeatureGroup.FeatureGroup(search_space, hot_encoded_features)
        self.cooccurrence_matrix = self.get_cooccurrence_matrix(candidate_matrix, fitness_list)
        self.diagonals = np.copy(np.diag(self.cooccurrence_matrix))
        self.maximum_value = sum(fitness_list)

    def score_of_feature_vector(self, feature_vector):
        row_matrix = feature_vector  # TODO check if feature_vector needs to be readapted into some form
        unnormalised_value = row_matrix @ self.cooccurrence_matrix @ np.transpose(row_matrix)  # a scalar, somehow
        return self.normalise_score_of_feature_vector(feature_vector, unnormalised_value)


    def normalise_score_of_feature_vector(self, feature_vector, unnormalised_value):
        amount_of_cells_involved = np.sum(feature_vector) ** 2
        if (amount_of_cells_involved == 0):
            print(f"WARNING: received a candidate with no features!")
            return 0
        average_value = unnormalised_value / amount_of_cells_involved
        return average_value / (self.maximum_value)

    def get_scores_of_own_features(self):
        # TODO perhaps i should divide them by the maximum in the diagonal
        return [diagonal_value / self.maximum_value for diagonal_value in
                self.diagonals]

    def score_of_raw_candidate_vector(self, raw_candidate_vector):
        as_features = self.feature_group.get_feature_vectors_for_candidates(
            raw_candidate_vector)  # a matrix, to be converted
        return self.score_of_feature_vector(as_features)



