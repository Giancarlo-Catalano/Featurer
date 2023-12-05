import copy
from collections import deque
from typing import Callable

from Version_E.BaselineAlgorithms import GA
from Version_E.BaselineAlgorithms.RandomSearch import random_feature_in_search_space
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, FeatureSelector


class HillClimber(FeatureMiner):
    chance_of_mutation = 0.075

    def __init__(self, selector: FeatureSelector, termination_criteria_met: Callable):
        super().__init__(selector, termination_criteria_met)

    def __repr__(self):
        return f"HillClimber()"

    def get_mutations_of_feature(self, feature: Feature) -> list[Feature]:
        amount_of_mutations = self.search_space.dimensions
        return [GA.mutate_feature(feature, self.search_space, self.chance_of_mutation) for _ in range(amount_of_mutations)]

    def get_improved_feature(self, feature: Feature) -> Feature:
        mutations = self.get_mutations_of_feature(feature)
        return self.feature_selector.keep_best_features(mutations, 1)[0][0]

    def improve_feature_until_stationary(self, feature: Feature) -> Feature:
        current_best_feature = copy.copy(feature)
        current_best_score = self.feature_selector.get_scores([current_best_feature])[0]

        def should_terminate():
            return self.termination_criteria_met(used_budget = self.feature_selector.used_budget)

        while not should_terminate():
            new_feature, new_score = self.get_improved_feature(current_best_feature)

            if new_score <= current_best_score:
                break

            current_best_feature = new_feature
            current_best_score = new_score


        return current_best_feature

    def get_meaningful_features(self, amount_to_return: int) -> list[Feature]:
        population = deque(random_feature_in_search_space(self.search_space)
                                     for _ in range(amount_to_return))

        iteration = 0

        def should_continue():
            return not self.termination_criteria_met(iteration=iteration,
                                                     returnable=population,
                                                     used_budget=self.feature_selector.used_budget)


        while should_continue():
            extracted = population.popleft()
            improved = self.get_improved_feature(extracted)
            population.append(improved)

        return list(population)
