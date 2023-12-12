import random
from typing import Iterable, Sized, Callable

import numpy as np

import utils
from SearchSpace import SearchSpace, Candidate

VariableDistribution = np.ndarray
Population = list[Candidate]
Fitness = float
EvaluatedPopulation = list[(Candidate, Fitness)]


class Evaluator:
    fitness_function: Callable
    used_evaluations: int

    def __init__(self, fitness_function: Callable):
        self.fitness_function = fitness_function
        self.used_evaluations = 0


    def __repr__(self):
        return "Evaluator"


    def evaluate(self, candidate: Candidate) -> Fitness:
        self.used_evaluations += 1
        return self.fitness_function(candidate)

    def with_fitnesses(self, population: Population) -> EvaluatedPopulation:
        """Calculates the fitnesses for each individual and places them in pairs alongside the individuals"""
        return [(candidate, self.fitness_function(candidate) for candidate in population)]

    def without_fitnesses(self, evaluated_population: EvaluatedPopulation) -> Population:
        """Removes the fitnesses and just returns a list of the individuals, in the same order as given"""
        if evaluated_population:
            return utils.unzip(evaluated_population)[0]
        else:
            return []

    def select(self, evaluated_population: EvaluatedPopulation, how_many: int) -> EvaluatedPopulation:
        """Finds the individuals with the greatest fitness and returns them without the fitness"""
        evaluated_population.sort(key=utils.second, reverse=True)
        return self.without_fitnesses(evaluated_population[:how_many])



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





