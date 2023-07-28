import itertools

import FeatureDiscoverer
import VariateModels
import utils
import SearchSpace
import HotEncoding
import numpy as np


class SurrogateScorer:
    model_power: int
    hot_encoder: HotEncoding.HotEncoder
    search_space: SearchSpace.SearchSpace

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
        self.features = featuresH

        self.variate_model_generator = VariateModels.VariateModels(self.search_space)
        self.redundant_cell_matrix = self.get_diagonal_cell_matrix(len(self.features), model_power)

        # these are filled during training
        self.S_matrix = None
        self.P_matrix = None


    def __repr(self):
        # in the future these things should come from the subclasses themselves
        rows_in_feature_detector = self.search_space.total_cardinality
        columns_in_feature_detector = len(self.features)



    def train(self, candidateC_list, fitness_list):
        """this will set the values for S and P"""

        # obtain which features are present in which candidates
        candidate_matrix = self.hot_encoder.to_hot_encoded_matrix(candidateC_list)
        feature_presence_matrix = self.variate_model_generator.get_feature_presence_matrix(candidate_matrix, self.features)

        # get self-interactions by using flat outer powers
        outer_power = utils.row_wise_nth_power_self_outer_product(feature_presence_matrix, self.model_power)

        # set S and P
        self.S_matrix = np.sum(outer_power * utils.to_column_vector(fitness_list),
                                         axis=0)
        self.P_matrix = np.sum(outer_power, axis=0)

    def get_surrogate_score_of_fitness(self, candidateC, picky=True):
        # obtain candidateH, place in a row matrix if necessary
        # obtain nth flat outer power of candidate H
        # sum_of_fitnesses = flat_outer dot_product self.S
        # sum_of_weights = flat_outer dot_product self.P
        # if picky, remove the diagonal entries
            # relevant_diagonal_cells = flat_outer * self.diagonal_cells  NOTE the normal multiplication
            # sum_of_diagonal_fitnesses = relevant_diagonal_cells dot_product self.S
            # sum_of_diagonal_weights = relevant_diagonal_cells dot_product self.P
            # sum_of_fitnesses -= sum_of_diagonal_fitnesses
            # sum_of_weights -= sum_of_diagonal_weights
        # if sum_of_weights == 0, return 0
        # return sum_of_fitnesses / sum_of_weights


        candidateH = self.hot_encoder.to_hot_encoding(candidateC)
        outer_power = utils.nth_power_flat_outer_product(candidateH, self.model_power)
        sum_of_fitnesses = np.dot(candidateH, self.S_matrix)
        sum_of_weights = np.dot(candidateH, self.P_matrix)
        if picky:
            relevant_redundant_cells = outer_power * self.redundant_cell_matrix
            sum_of_redundant_fitnesses = np.dot(relevant_redundant_cells, self.S_matrix)
            sum_of_redundant_weights = np.dot(relevant_redundant_cells, self.P_matrix)
            sum_of_fitnesses -= sum_of_redundant_fitnesses
            sum_of_weights   -= sum_of_redundant_weights

        if sum_of_weights == 0.0:
            return 0.0
        return sum_of_fitnesses/sum_of_weights



# TODO make a CandidateValidator class, and a FeatureDetector class