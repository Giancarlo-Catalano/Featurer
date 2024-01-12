import heapq
import random
from typing import Callable, Optional
import numpy as np
import SearchSpace
import utils
from SearchSpace import Candidate
from Version_E.Sampling.FullSolutionSampler import FullSolutionSampler, Population, EvaluatedPopulation, Fitness


class GASampler(FullSolutionSampler):
    population_size: int
    used_generations: int

    def __init__(self,
                 fitness_function: Callable,
                 search_space: SearchSpace.SearchSpace,
                 population_size: int,
                 termination_criteria: Callable):
        super().__init__(search_space, fitness_function, termination_criteria)
        self.population_size = population_size
        self.used_generations = 0

    def __repr__(self):
        return "GASampler"


    def generate_selector(self, evaluated_population: EvaluatedPopulation):  # -> SupportsNext[Candidate]
        tournament_size = 2
        sorted_population, _ = utils.unzip(sorted(evaluated_population, key=utils.second))
        population_size = len(sorted_population)

        while True:
            tournament_indexes = [random.randrange(population_size) for _ in range(tournament_size)]
            winning_index = max(tournament_indexes)
            yield sorted_population[winning_index]



    def generate_selector_old(self, evaluated_population: EvaluatedPopulation):  # -> SupportsNext[Candidate]:
        population, scores = utils.unzip(evaluated_population)

        normalised_scores = utils.remap_array_in_zero_one(scores)
        normalised_scores = normalised_scores / np.sum(normalised_scores)
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

        self.used_generations += 1
        return children

    def evolve_population(self, population=None) -> Population:
        if population is None:
            population = self.get_random_population(self.population_size)

        while not self.termination_criteria(used_budget=self.evaluator.used_evaluations, iteration=self.used_generations):
            population = self.get_new_generation(population)

        return population

    def get_final_population(self) -> Population:
        return self.evolve_population(population=None)