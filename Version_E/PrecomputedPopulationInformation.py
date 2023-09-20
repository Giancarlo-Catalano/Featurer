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
                 population_sample: list[SearchSpace.Candidate],
                 fitness_list: list[float]):
        self.search_space = search_space
        self.candidate_matrix = HotEncoding.hot_encode_candidate_population(population_sample, self.search_space)
        self.fitness_array = np.array(fitness_list)
        self.sample_size = len(population_sample)

    @classmethod
    def from_problem(cls, problem: CombinatorialProblem.CombinatorialProblem,
                     amount_of_samples: int):
        search_space = problem.search_space
        population = [problem.get_random_candidate_solution() for _ in range(amount_of_samples)]
        fitnesses = [problem.score_of_candidate(candidate) for candidate in population]
        return cls(search_space, population, fitnesses)

    def count_for_each_var_val(self) -> list[list[float]]:
        sum_in_hot_encoding_order: list[float] = np.sum(self.candidate_matrix, axis=0).tolist()
        def counts_for_each_variable(var_index):
            start, end = self.search_space.precomputed_offsets[var_index: var_index+2]
            return sum_in_hot_encoding_order[start:end]

        return [counts_for_each_variable(var_index) for var_index in range(self.search_space.dimensions)]