import itertools

import utils
import numpy as np
import ProgressiveFeatures
import CooccurrenceModel
import SearchSpace
import copy


class SimpleVariableExplainer:
    cooccurrence_model: CooccurrenceModel.CooccurrenceModel
    search_space: SearchSpace.SearchSpace

    def __init__(self, progressive_features_before_build: ProgressiveFeatures.ProgressiveFeatures):
        self.cooccurrence_model = copy.copy(progressive_features_before_build.cooccurrence_model)
        self.search_space = progressive_features_before_build.search_space

    def get_binomial_frequencies(self, var_index_a, var_index_b):
        def start_and_end_of_var(index):
            return (self.search_space.precomputed_offsets[index], self.search_space.precomputed_offsets[index+1])

        (start_a, end_a) = start_and_end_of_var(var_index_a)
        (start_b, end_b) = start_and_end_of_var(var_index_b)

        return self.cooccurrence_model.cooccurrence_matrix[start_a:end_a, start_b:end_b]

    def is_chi_squared_significant_95(self, chi_squared, degrees_of_freedom):
        pearson_95_thresholds = [0, 3.841, 5.991, 7.815, 9.488, 11.07, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026, 22.362, 23.685, 24.996, 26.296, 27.587, 28.869, 30.144, 31.41, 32.671, 33.924, 35.172, 36.415, 37.652, 38.885, 40.113, 41.337, 42.557, 43.773, 55.758, 67.505, 79.082, 90.531, 101.879, 113.145, 124.342]
        if (degrees_of_freedom  > len(pearson_95_thresholds)):
            print("Warning, the amount of degrees of freedom exceeds what is available in out Chi-squared table")
            return True
        return chi_squared >= pearson_95_thresholds[degrees_of_freedom]

    def get_correlation_of_variables(self, var_index_a, var_index_b):
        binomial_frequencies = self.get_binomial_frequencies(var_index_a, var_index_b) / self.cooccurrence_model.maximum_value
        marginal_frequencies_a = np.sum(binomial_frequencies, axis=0)
        marginal_frequencies_b = np.sum(binomial_frequencies, axis=1)

        expected_frequencies = utils.to_column_vector(marginal_frequencies_a) @ utils.array_to_row_vector(
            marginal_frequencies_b)


        binomial_frequencies *= self.cooccurrence_model.maximum_value
        expected_frequencies *= self.cooccurrence_model.maximum_value

        total_chi_squared = sum([utils.chi_squared(observed, expected)
                              for (observed, expected)
                                  in zip(np.nditer(binomial_frequencies), np.nditer(expected_frequencies))])

        return total_chi_squared

    def are_variables_correlated(self, var_index_a, var_index_b):
        total_chi_squared = self.get_correlation_of_variables(var_index_a, var_index_b)

        cardinality_a = self.search_space.cardinalities[var_index_a]
        cardinality_b = self.search_space.cardinalities[var_index_b]
        degrees_of_freedom = (cardinality_a-1) * (cardinality_b-1)
        return self.is_chi_squared_significant_95(total_chi_squared, degrees_of_freedom)

    def get_non_uniformness_of_variable(self, variable_index):
        variable_self_correlation_values = np.diag(self.get_binomial_frequencies(variable_index, variable_index))

        def jump_in_fitness(val_x, val_y):
            return abs(variable_self_correlation_values[val_x]-variable_self_correlation_values[val_y])


        cardinality = self.search_space.cardinalities[variable_index]
        total_jump_in_fitness = sum([jump_in_fitness(x, y) for (x, y) in itertools.combinations(range(cardinality), 2)])
        average_jump_in_fitness = total_jump_in_fitness / utils.binomial_coeff(cardinality, 2)
        return average_jump_in_fitness


    def print_correlation_report(self):
        variable_index_list = list(range(self.search_space.dimensions))
        variable_correlation_matrix = np.array([[self.get_correlation_of_variables(var_a, var_b)
                                        for var_b in variable_index_list]
                                        for var_a in variable_index_list])

        variable_correlation_matrix_boolean = np.array([[int(self.are_variables_correlated(var_a, var_b))
                                        for var_b in variable_index_list]
                                        for var_a in variable_index_list])

        importance_of_variables = [(var_index, self.get_non_uniformness_of_variable(var_index))
                                   for var_index in variable_index_list]

        def pretty_print_matrix(matrix):
            for row in matrix:
                for cell in row:
                    print(f"\t{cell:.2f}", end="")
                print()
            print()

        print(f"For the given problem, the correlation matrix is")
        pretty_print_matrix(variable_correlation_matrix)

    def get_variance_of_variable(self, variable_index):
        variable_self_correlation_values = np.diag(self.get_binomial_frequencies(variable_index, variable_index))
        n = len(variable_self_correlation_values)
        mean = sum(variable_self_correlation_values) / n

        return sum([(value-mean)**2 for value in variable_self_correlation_values])/n


    def get_importance_candidate(self):
        variable_index_list = range(self.search_space.dimensions)
        importance_of_variables = [self.get_variance_of_variable(var_index)
                                   for var_index in variable_index_list]
        return SearchSpace.Candidate(importance_of_variables)
