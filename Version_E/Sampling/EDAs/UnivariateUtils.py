import random
from typing import Iterable

import numpy as np

from SearchSpace import SearchSpace, Candidate
from Version_E.Sampling.FullSolutionSampler import Population

VariableDistribution = np.ndarray


class UnivariateModel:
    search_space: SearchSpace
    probabilities: list[VariableDistribution]

    def __init__(self, probabilities: Iterable[VariableDistribution], search_space: SearchSpace):
        self.probabilities = list(probabilities)
        self.search_space = search_space

    def __repr__(self):
        def repr_cell(cell: VariableDistribution):
            return "[" + "|".join(f"{value}:.2f" for value in cell) + "]"

        return " ".join(map(repr_cell, self.probabilities))

    @classmethod
    def get_uniform(cls, search_space: SearchSpace):
        """Returns the uniform distribution model for a given search space"""
        def get_uniform_cell(var_index: int) -> VariableDistribution:
            cardinality = search_space.cardinalities[var_index]
            return np.ones(cardinality, dtype=float) / cardinality

        return cls(map(get_uniform_cell, range(search_space.dimensions)), search_space)

    @classmethod
    def get_from_selected_population(cls, population: Population, search_space: SearchSpace):
        """Returns the model obtained by looking at a given distribution"""
        def get_variable_distribution(var_index):
            cardinality = search_space.cardinalities[var_index]
            counter = np.zeros(cardinality, dtype=float)
            for candidate in population:
                observed_value = candidate.values[var_index]
                counter[observed_value] += 1
            return counter / len(population)

        return cls(map(get_variable_distribution, range(search_space.dimensions)), search_space)

    @classmethod
    def weighted_sum(cls, a, b, weight_a: float, weight_b: float):
        """Combines two models into one using a weighted sum (similar to PBIL)"""
        sum_of_weights = weight_a + weight_b

        def weighted_avg_of_distributions(distr_a: VariableDistribution,
                                          distr_b: VariableDistribution) -> VariableDistribution:
            return (distr_a * weight_a + distr_b * weight_b) / sum_of_weights

        return cls(map(weighted_avg_of_distributions, a.probabilities, b.probabilities), a.search_space)


    def sample(self) -> Candidate:
        """Returns a new candidate with the univariate distribution described in self.probabilities"""
        def sample_variable(var_index: int) -> int:
            options = list(range(self.search_space.cardinalities[var_index]))
            weights = self.probabilities[var_index]
            return random.choices(options, weights=weights, k=1)[0]

        return Candidate(tuple(map(sample_variable, range(self.search_space.dimensions))))

    def sample_many(self, how_many: int) -> list[Candidate]:
        return [self.sample() for _ in range(how_many)]





