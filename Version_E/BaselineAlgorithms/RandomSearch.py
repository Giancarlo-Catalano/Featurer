import random
from typing import Callable

import numpy as np
from bitarray import frozenbitarray
from bitarray.util import urandom

import utils
from SearchSpace import SearchSpace
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureSelector, FeatureMiner


def random_feature_in_search_space(search_space: SearchSpace) -> Feature:
    used_cells = frozenbitarray(urandom(search_space.dimensions))
    values = np.zeros(search_space.dimensions, dtype=int)
    for variable, is_used in enumerate(used_cells):
        if is_used:
            values[variable] = random.randrange(search_space.cardinalities[variable])

    return Feature(used_cells, values)


class RandomSearch(FeatureMiner):
    amount_to_generate: int

    def __init__(self, selector: FeatureSelector, termination_criteria_met: Callable):
        super().__init__(selector, termination_criteria_met)

    def __repr__(self):
        return f"RandomSearch()"

    def get_random_feature(self) -> Feature:
        return random_feature_in_search_space(self.search_space)

    def get_meaningful_features(self, amount_to_return: int) -> list[Feature]:
        batch_size = amount_to_return

        def generate_batch_of_random_features() -> list[Feature]:
            return [self.get_random_feature() for _ in range(batch_size)]

        population = []
        iteration = 0

        def should_continue():
            return not self.termination_criteria_met(iteration=iteration,
                                                     population=population,
                                                     archive=set(),
                                                     used_budget=self.feature_selector.used_budget)

        while should_continue():
            population.extend(generate_batch_of_random_features())
            evaluated_population = self.with_scores(population)
            evaluated_population = self.truncation_selection(evaluated_population, amount_to_return)
            population = self.without_scores(evaluated_population)
            iteration += 1

        return population
