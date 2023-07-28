import itertools

import VariateModels
import utils
import SearchSpace
import HotEncoding
import numpy as np


class SurrogateScorer:
    model_power: int
    hot_encoder: HotEncoding.HotEncoder
    search_space: SearchSpace.SearchSpace
    feature_detector: VariateModels.FeatureDetector

    #features
    #sum_of_scores_matrix
    #sum_of_presence_matrix


    def get_diagonal_cell_matrix(self, rows, power):
        """returns a flat matrix"""
        # if power is 2, it's the identity matrix
        # generally, it's the power-dimensional matrix where
        # (D)_{{i_0}, {i_1},\ldots,{i_n}} = 1 if any of the i_n values are the same

        if power == 2:
            return np.identity(rows).ravel()

        def contains_overlap(list_of_indices):
            seen_already = [False]*rows
            for index in list_of_indices:
                if seen_already[index]:
                    return True
                seen_already = True
            return False


        result_as_list = [float(contains_overlap(list_of_indices))
                         for list_of_indices in itertools.combinations_with_replacement(range(rows), power)]

        return np.array(result_as_list, dtype=np.float)

    def __init__(self, model_power, search_space, featuresH):
        self.model_power = model_power
        self.search_space = search_space
        self.feature_detector = VariateModels.FeatureDetector(search_space, featuresH)

        self.variate_model_generator = VariateModels.VariateModels(self.search_space)
        self.redundant_cell_matrix = self.get_diagonal_cell_matrix(len(featuresH), model_power)

        # these are filled during training
        self.S_matrix = None
        self.P_matrix = None


    def __repr__(self):
        # in the future these things should come from the subclasses themselves
        result = f"SurrogateScorer: \n\t{self.feature_detector.__repr__()}"
        if self.S_matrix is None:
            result += "\n\tUntrained"
        else:
            (s_rows, s_cols) = self.S_matrix.shape
            (p_rows, p_cols) = self.P_matrix.shape
            result += f"\n\tTrained, with S: {s_rows} x {s_cols}, P: {p_rows} x {p_cols}"

        return result

    def train(self, candidatesC, fitness_list):
        """this will set the values for S and P"""

        # obtain which features are present in which candidates
        feature_presence_matrix = self.feature_detector.get_feature_presence_matrix_from_candidates(candidatesC)

        # get self-interactions by using flat outer powers
        outer_power = utils.row_wise_nth_power_self_outer_product(feature_presence_matrix, self.model_power)

        # set S and P
        self.S_matrix = np.sum(outer_power * utils.to_column_vector(fitness_list), axis=0)
        self.P_matrix = np.sum(outer_power, axis=0)

    def get_surrogate_score_of_fitness(self, candidateC, picky=True):
        candidate_feature_vector = self.feature_detector.get_feature_presence_matrix_from_candidates(candidateC)
        outer_power = utils.nth_power_flat_outer_product(candidate_feature_vector, self.model_power)
        sum_of_fitnesses = np.dot(candidate_feature_vector, self.S_matrix)
        sum_of_weights = np.dot(candidate_feature_vector, self.P_matrix)
        if picky:
            relevant_redundant_cells = outer_power * self.redundant_cell_matrix
            sum_of_redundant_fitnesses = np.dot(relevant_redundant_cells, self.S_matrix)
            sum_of_redundant_weights = np.dot(relevant_redundant_cells, self.P_matrix)
            sum_of_fitnesses -= sum_of_redundant_fitnesses
            sum_of_weights   -= sum_of_redundant_weights

        if sum_of_weights == 0.0:
            return 0.0
        return sum_of_fitnesses/sum_of_weights
