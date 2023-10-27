import copy
from typing import Callable

from Version_E.BaselineAlgorithms.RandomSearch import random_feature_in_search_space
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, FeatureSelector


class HillClimber(FeatureMiner):

    def __init__(self, selector: FeatureSelector, termination_criteria_met: Callable):
        super().__init__(selector, termination_criteria_met)

    def __repr__(self):
        return f"HillClimber()"

    def get_mutations_of_feature(self, feature: Feature) -> list[Feature]:
        return (feature.get_generalisations() +
                feature.get_specialisations(self.search_space) +
                feature.get_variations(self.search_space))

    def get_improved_feature(self, feature: Feature) -> Feature:
        mutations = self.get_mutations_of_feature(feature)
        return self.feature_selector.keep_best_features(mutations, 1)[0][0]

    def improve_feature_until_stationary(self, feature: Feature) -> Feature:
        current_best_feature = copy.copy(feature)
        current_best_score = self.feature_selector.get_scores([current_best_feature])[0]

        while True:
            new_feature, new_score = self.get_improved_feature(current_best_feature)
            if new_score > current_best_score:
                current_best_feature = new_feature
                current_best_score = new_score
            else:
                break

        return current_best_feature

    def get_meaningful_features(self, amount_to_return: int) -> list[Feature]:
        population: list[Feature] = [random_feature_in_search_space(self.search_space)
                                     for _ in range(amount_to_return)]

        iteration = 0

        def should_continue():
            return not self.termination_criteria_met(iteration=iteration,
                                                     population=population,
                                                     archive=set(),
                                                     used_budget=self.feature_selector.used_budget)

        while should_continue():
            population = [self.get_improved_feature(individual) for individual in population]
            iteration +=1

        return population
