import copy
import random
from typing import Callable

import numpy as np
from bitarray import bitarray, frozenbitarray

import SearchSpace
import utils
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, FeatureSelector
from Version_E.BaselineAlgorithms.RandomSearch import random_feature_in_search_space
import heapq


def mutate_feature(feature: Feature,
           search_space: SearchSpace.SearchSpace,
           chance_of_mutation: float) -> Feature:
    def with_empty(current: Feature, var_index) -> Feature:
        return current.with_value(var_index, None)

    def with_random_value(current: Feature, var_index) -> Feature:
        available_values = list(range(search_space.cardinalities[var_index]))
        if current.variable_mask[var_index]:
            available_values.pop(current.values_mask[var_index])
        return current.with_value(var_index, random.choice(available_values))

    mutated: Feature = copy.copy(feature)
    for var_index in range(search_space.dimensions):
        if random.random() < chance_of_mutation:
            if mutated.variable_mask[var_index] and random.random() < 0.5:
                mutated = with_empty(mutated, var_index)
            else:
                mutated = with_random_value(mutated, var_index)

    return mutated

class GAMiner(FeatureMiner):
    population_size: int

    tournament_size = 2
    chance_of_mutation = 0.075
    chance_of_crossover = 0.7
    elite_size = 2

    Population = list[Feature]
    EvaluatedPopulation = list[(Feature, float)]

    def __init__(self, selector: FeatureSelector, population_size: int, termination_criteria_met: Callable, crossover_enabled=True):
        super().__init__(selector, termination_criteria_met)
        self.population_size = population_size
        if not crossover_enabled:
            self.chance_of_crossover = 0

    def __repr__(self):
        return f"GA(population = {self.population_size})"

    def mutate(self, feature: Feature) -> Feature:
        return mutate_feature(feature, self.search_space, self.chance_of_mutation)

    def crossover(self, mother: Feature, father: Feature):
        crossover_point = random.randrange(self.search_space.dimensions)

        result_variable_mask = bitarray(mother.variable_mask[:crossover_point])
        result_variable_mask.extend(father.variable_mask[crossover_point:])
        result_value_array = np.concatenate((mother.values_mask[:crossover_point],
                                             father.values_mask[crossover_point:]))

        return Feature(frozenbitarray(result_variable_mask), result_value_array)


    def tournament_select(self, population: EvaluatedPopulation) -> Feature:
        tournament_pool = random.choices(population, k=self.tournament_size)
        winner = max(tournament_pool, key=utils.second)[0]
        return winner

    def with_scores(self, features) -> list[(Feature, float)]:
        scores = self.feature_selector.get_scores(features)
        return list(zip(features, scores))

    def get_initial_population(self) -> list[(Feature, float)]:
        individuals = [random_feature_in_search_space(self.search_space) for _ in range(self.population_size)]
        return self.with_scores(individuals)


    def get_elite(self, evaluated_population: EvaluatedPopulation) -> Population:
        top_evaluated = heapq.nlargest(self.elite_size, evaluated_population, key=utils.second)
        if len(top_evaluated) == 0:
            return []
        else:
            return self.without_scores(top_evaluated)

    def get_next_population(self, previous_population: Population) -> Population:

        evaluated_population = self.with_scores(previous_population)

        def make_child() -> Feature:
            if random.random() < self.chance_of_crossover:
                mother = self.tournament_select(evaluated_population)
                father = self.tournament_select(evaluated_population)
                child = self.crossover(mother, father)
                child = self.mutate(child)
                return child
            else:
                return self.mutate(self.tournament_select(evaluated_population))

        children = self.get_elite(evaluated_population)
        remaining_to_generate = self.population_size - len(children)
        children = [make_child() for _ in range(remaining_to_generate)]
        return children

    def mine_features(self) -> list[Feature]:
        population = [random_feature_in_search_space(self.search_space) for _ in range(self.population_size)]

        iteration = 0

        def should_continue():
            return not self.termination_criteria_met(iteration=iteration,
                                                     returnable=population,
                                                     used_budget=self.feature_selector.used_budget)

        while should_continue():
            population = self.get_next_population(population)
            iteration += 1

        return population
