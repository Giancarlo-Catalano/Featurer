import copy
import random

import numpy as np
from bitarray import bitarray, frozenbitarray

import utils
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, FeatureSelector
from RandomSearch import random_feature_in_search_space


class GAMiner(FeatureMiner):
    population_size: int
    iterations: int

    tournament_size = 30
    chance_of_mutation = 0.05

    def __init__(self, selector: FeatureSelector, population_size: int, iterations: int):
        super().__init__(selector)
        self.population_size = population_size
        self.iterations = iterations

    def mutate(self, feature: Feature) -> Feature:
        def with_empty(current: Feature, var_index) -> Feature:
            return current.with_value(var_index, None)

        def with_random_value(current: Feature, var_index) -> Feature:
            available_values = list(range(self.search_space.cardinalities[var_index]))
            if current.variable_mask[var_index]:
                available_values.remove(current.values_mask[var_index])
            return current.with_value(var_index, random.choice(available_values))

        def mutate_cell(current: Feature, var_index) -> None:
            if random.random() < self.chance_of_mutation:
                if current.variable_mask[var_index] and random.random() < 0.5:
                    current = with_empty(current, var_index)
                else:
                    current = with_random_value(current, var_index)

        mutated: Feature = copy.copy(feature)
        for var_index in range(self.search_space.dimensions):
            mutate_cell(mutated, var_index)

        return mutated

    def crossover(self, mother: Feature, father: Feature):
        crossover_point = random.randrange(self.search_space.dimensions)

        result_variable_mask = mother.variable_mask[:crossover_point]
        result_variable_mask.extend(father.variable_mask[crossover_point:])
        result_value_array = np.concatenate(mother.values_mask[:crossover_point],
                                            father.values_mask[crossover_point:])

        return Feature(frozenbitarray(result_variable_mask), result_value_array)
    def tournament_select(self, population: list[(Feature, float)]) -> Feature:
        tournament_pool = random.choices(population, k=self.tournament_size)
        tournament_pool.sort(key=utils.second, reverse=True)
        winner, score_of_winner = tournament_pool[0]
        return winner

    def with_scores(self, features) -> list[(Feature, float)]:
        scores = self.feature_selector.get_scores(features)
        return list(zip(features, scores))
    def get_initial_population(self) -> list[(Feature, float)]:
        individuals = [random_feature_in_search_space(self.search_space) for _ in range(self.population_size)]
        return self.with_scores(individuals)

    def get_next_population(self, previous_population: list[(Feature, float)]):
        def make_child() -> Feature:
            mother = self.tournament_select(previous_population)
            father = self.tournament_select(previous_population)
            return self.mutate(self.crossover(mother, father))

        return self.with_scores([make_child() for _ in range(self.population_size)])

    def mine_features(self) -> list[Feature]:
        population = self.get_initial_population()
        for iteration in range(self.iterations):
            population = self.get_next_population(population)

        return population


