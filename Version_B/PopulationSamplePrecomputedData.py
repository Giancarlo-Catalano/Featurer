import SearchSpace
import numpy as np
import HotEncoding
from BenchmarkProblems import CombinatorialProblem
from Version_B import VariateModels
import utils


class PopulationSamplePrecomputedData:
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


class PopulationSampleWithFeaturesPrecomputedData:
    """this data structures stores matrices that are used around the other classes"""
    population_sample_precomputed: PopulationSamplePrecomputedData
    feature_presence_matrix: np.ndarray

    count_for_each_feature: np.ndarray
    complexity_array: np.ndarray

    def __init__(self, population_precomputed: PopulationSamplePrecomputedData,
                 raw_features: list[SearchSpace.Feature]):
        self.population_sample_precomputed = population_precomputed
        feature_matrix = HotEncoding.hot_encode_feature_list(raw_features,
                                                             self.population_sample_precomputed.search_space)

        self.feature_presence_matrix = VariateModels.get_feature_presence_matrix_from_feature_matrix(
            self.population_sample_precomputed.candidate_matrix,
            feature_matrix)

        self.count_for_each_feature = np.sum(self.feature_presence_matrix, axis=0)

    def get_average_fitness_vector(self) -> np.ndarray:
        """returns the vector of average fitnesses for each feature"""
        sum_of_fitnesses = utils.weighted_sum_of_rows(self.feature_presence_matrix,
                                                      self.population_sample_precomputed.fitness_array)

        return utils.divide_arrays_safely(sum_of_fitnesses, self.count_for_each_feature)

    def get_overall_average_fitness(self):
        """returns the average fitness over the entire population"""
        return np.mean(self.population_sample_precomputed.fitness_array)

    def get_observed_proportions(self):
        """returns the observed proportion for every feature, from 0 to 1"""
        return self.count_for_each_feature / self.population_sample_precomputed.sample_size
