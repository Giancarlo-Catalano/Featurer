from typing import Callable

import utils
from SearchSpace import Candidate, SearchSpace
from Version_E.Feature import Feature

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
        return [(candidate, self.fitness_function(candidate)) for candidate in population]

    def without_fitnesses(self, evaluated_population: EvaluatedPopulation) -> Population:
        """Removes the fitnesses and just returns a list of the individuals, in the same order as given"""
        if evaluated_population:
            return utils.unzip(evaluated_population)[0]
        else:
            return []

    def select(self, evaluated_population: EvaluatedPopulation, how_many: int) -> EvaluatedPopulation:
        """Finds the individuals with the greatest fitness and returns them without the fitness"""
        """Truncation selection basically"""
        evaluated_population.sort(key=utils.second, reverse=True)
        return self.without_fitnesses(evaluated_population[:how_many])


    def get_best(self, population: Population, how_many_to_return) -> EvaluatedPopulation:
        evaluated = self.with_fitnesses(population)
        evaluated.sort(key=utils.second, reverse=True)
        return evaluated[:how_many_to_return]


class FullSolutionSampler:
    search_space: SearchSpace
    evaluator: Evaluator
    termination_criteria: Callable

    def __init__(self, search_space: SearchSpace,
                 fitness_function: Callable,
                 termination_criteria: Callable):
        self.search_space = search_space
        self.evaluator = Evaluator(fitness_function)
        self.termination_criteria = termination_criteria

    def get_final_population(self) -> Population:
        raise Exception("An implementation of FullSolutionSampler did not implement .sample")

    def with_scores(self, population: Population) -> EvaluatedPopulation:
        return [(candidate, self.evaluator.evaluate(candidate)) for candidate in population]


    def without_scores(self, evaluated_population: EvaluatedPopulation) -> Population:
        if len(evaluated_population) == 0:
            return []
        else:
            return utils.unzip(evaluated_population)[0]

    def sample(self, how_many_to_return: int) -> EvaluatedPopulation:
        return self.evaluator.get_best(self.get_final_population(), how_many_to_return)


    def reset_used_budget(self):
        self.evaluator.used_evaluations = 0

    def get_random_population(self, amount_to_generate: int) -> Population:
        return [self.search_space.get_random_candidate()
                for _ in range(amount_to_generate)]
