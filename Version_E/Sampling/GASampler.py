import heapq
import random
from typing import Callable, Optional
import numpy as np
import SearchSpace
import utils
from SearchSpace import Candidate


class GASampler:
    Population = list[Candidate]
    EvaluatedPopulation = list[(Candidate, float)]

    fitness_function: Callable
    search_space: SearchSpace.SearchSpace
    population_size: int
    evaluation_budget: int

    used_budget: int

    def __init__(self, fitness_function: Callable, search_space: SearchSpace.SearchSpace, population_size: int,
                 termination_criteria: Callable):
        self.fitness_function = fitness_function
        self.search_space = search_space
        self.population_size = population_size
        self.termination_criteria = termination_criteria
        self.used_budget = 0

    def __repr__(self):
        return "GASampler"

    def evaluate_individual(self, candidate: Candidate) -> float:
        self.used_budget += 1
        return self.fitness_function(candidate)

    def with_scores(self, population: Population) -> EvaluatedPopulation:
        return [(candidate, self.evaluate_individual(candidate)) for candidate in population]

    def generate_selector(self, evaluated_population: EvaluatedPopulation):  # -> SupportsNext[Candidate]:
        population, scores = utils.unzip(evaluated_population)

        normalised_scores = utils.remap_array_in_zero_one(scores)
        cumulative_weights = np.cumsum(normalised_scores)

        batch_size = len(evaluated_population) * 2

        def generate_batch() -> list[Candidate]:
            return random.choices(population, cumulative_weights, k=batch_size)

        batch_index = 0
        current_batch = generate_batch()

        while True:
            yield current_batch[batch_index]
            batch_index += 1
            if batch_index >= batch_size:
                current_batch = generate_batch()
                batch_index = 0

    def mutated(self, candidate: Candidate) -> Candidate:
        chance_of_mutation = 0.05  # per cell
        result_values = np.array(candidate.values)

        for i, _ in enumerate(candidate.values):
            if random.random() < chance_of_mutation:
                random_value = random.choice(list(range(self.search_space.cardinalities[i])))
                result_values[i] = random_value

        return Candidate(result_values)

    def crossover(self, mother: Candidate, father: Candidate) -> Candidate:
        # standard 2 point crossover
        last_index = len(mother.values)
        start_cut = random.randrange(last_index)
        end_cut = random.randrange(last_index)
        start_cut, end_cut = min(start_cut, end_cut), max(start_cut, end_cut)

        def take_from(donor: Candidate, start_index, end_index) -> list[int]:
            return list(donor.values[start_index:end_index])

        child_value_list = (take_from(mother, 0, start_cut) +
                            take_from(father, start_cut, end_cut) +
                            take_from(mother, end_cut, last_index))

        return Candidate(child_value_list)

    def get_new_generation(self, population: Population) -> Population:
        elite_size = 2
        evaluated_population = self.with_scores(population)
        elite = utils.unzip(heapq.nlargest(elite_size, evaluated_population, key=utils.second))[0]
        selector = self.generate_selector(evaluated_population)

        def make_new_child():
            return self.mutated(self.crossover(next(selector), next(selector)))

        children = list(elite)
        children.extend(make_new_child() for _ in range(self.population_size - elite_size))

        return children

    def reset_used_budget(self):
        self.used_budget = 0

    def evolve_population(self, population=None) -> Population:
        if population is None:
            population = [self.search_space.get_random_candidate()
                          for _ in range(self.population_size)]

        while not self.termination_criteria(used_budget=self.used_budget):
            population = self.get_new_generation(population)

        return population

    def get_evolved_individuals(self, amount_requested: int) -> Population:
        result = []
        while len(result) < amount_requested:
            result.extend(self.evolve_population())

        return result[:amount_requested]
