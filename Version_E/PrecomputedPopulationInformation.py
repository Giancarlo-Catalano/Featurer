from copy import copy

import SearchSpace
import numpy as np
from Version_E import HotEncoding
from BenchmarkProblems import CombinatorialProblem

class PrecomputedPopulationInformation:
    search_space: SearchSpace.SearchSpace
    candidate_matrix: np.ndarray
    fitness_array: np.ndarray
    sample_size: int

    def __init__(self, search_space: SearchSpace.SearchSpace,
                 population_sample,  # : list[SearchSpace.Candidate],
                 fitness_list: list[float],
                 population_is_full_solutions = True):
        self.search_space = search_space
        if population_is_full_solutions:  # this is the most common behaviour
            self.candidate_matrix = HotEncoding.hot_encode_candidate_population(population_sample, self.search_space)
        else:  # this is just a shortcut for sampling
            self.candidate_matrix = np.array([HotEncoding.get_hot_encoded_feature(feature, self.search_space)
                                              for feature in population_sample])
        self.fitness_array = np.array(fitness_list)
        self.sample_size = len(population_sample)

    def __repr__(self):
        return f"PPI(sample_size = {self.sample_size})"

    @classmethod
    def from_problem(cls, problem: CombinatorialProblem.CombinatorialProblem,
                     amount_of_samples: int):
        search_space = problem.search_space
        population = [problem.get_random_candidate_solution() for _ in range(amount_of_samples)]
        fitnesses = [problem.score_of_candidate(candidate) for candidate in population]
        return cls(search_space, population, fitnesses)

    @classmethod
    def from_preevolved_population(cls, preevolved_population,
                                   problem: CombinatorialProblem.CombinatorialProblem):
        fitness_list = [problem.score_of_candidate(candidate) for candidate in preevolved_population]
        return cls(problem.search_space, preevolved_population, fitness_list)


    def get_reduced_version(self, new_sample_size: int):
        # perhaps this should be randomised?
        result = copy(self)
        if self.sample_size <= new_sample_size:
            return result  #no cutting required
        result.sample_size = new_sample_size
        result.candidate_matrix = result.candidate_matrix[:new_sample_size]
        result.fitness_array = result.fitness_array[:new_sample_size]
        return result

    def count_for_each_var_val(self) -> list[list[float]]:
        sum_in_hot_encoding_order: list[float] = np.sum(self.candidate_matrix, axis=0).tolist()
        def counts_for_each_variable(var_index):
            start, end = self.search_space.precomputed_offsets[var_index: var_index+2]
            return sum_in_hot_encoding_order[start:end]

        return [counts_for_each_variable(var_index) for var_index in range(self.search_space.dimensions)]