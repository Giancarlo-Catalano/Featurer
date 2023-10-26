import random
from typing import Iterable, Callable

import numpy as np

import utils
from Version_E import HotEncoding
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, FeatureSelector

Score = float


class GCMiner(FeatureMiner):
    population_size: int
    uses_archive: bool

    Population = list[Feature]
    EvaluatedPopulation = list[(Feature, float)]

    def __init__(self,
                 selector: FeatureSelector,
                 population_size: int,
                 uses_archive: bool,
                 termination_criteria_met: Callable):
        super().__init__(selector, termination_criteria_met)
        self.population_size = population_size
        self.uses_archive = uses_archive
        self.termination_criteria_met = termination_criteria_met

    def __repr__(self):
        return f"ArchiveMiner(population = {self.population_size})"





    def without_features_in_archive(self, population: Population, archive: set[Feature]) -> Population:
        return [feature for feature in population if feature not in archive]

    def get_empty_feature_population(self) -> Population:
        return [Feature.empty_feature(self.search_space)]

    def get_complex_feature_population(self, amount_to_return) -> Population:
        fitnesses_and_indexes = list(enumerate(self.feature_selector.ppi.fitness_array))
        fitnesses_and_indexes = sorted(fitnesses_and_indexes, key=utils.second, reverse=True)
        good_individuals = fitnesses_and_indexes[:amount_to_return]
        indexes_of_good_individuals = utils.unzip(good_individuals)[0]

        def get_individual(index: int):
            hot_encoded_feature = self.feature_selector.ppi.candidate_matrix[index]
            return HotEncoding.feature_from_hot_encoding(hot_encoded_feature, self.search_space)

        return [get_individual(index) for index in indexes_of_good_individuals]

    def get_complected_children(self, parents: EvaluatedPopulation) -> list[Feature]:
        return [child for parent, score in parents for child in Feature.get_specialisations(parent, self.search_space)]

    def get_simplified_children(self, parents: EvaluatedPopulation) -> list[Feature]:
        return [child for parent, score in parents for child in Feature.get_generalisations(parent)]

    def get_initial_population(self) -> Population:
        raise Exception("An implementation of ArchiveMiner does not implement get_initial_population")

    def get_children(self, parents: EvaluatedPopulation) -> list[Feature]:
        raise Exception("An implementation of ArchiveMiner does not implement get_children")

    def select(self, population: EvaluatedPopulation) -> EvaluatedPopulation:
        raise Exception("An implementation of ArchiveMiner does not implement select")

    def mine_features_with_archive(self) -> Population:
        population = self.get_initial_population()
        archive = set()
        iteration = 0

        def should_continue():
            return (not self.termination_criteria_met(iteration=iteration,
                                                      population=population,
                                                      archive=archive,
                                                      used_budget=self.feature_selector.used_budget)) and len(
                population) > 0

        while should_continue():
            iteration += 1
            print(f"In iteration {iteration}")
            population = self.remove_duplicate_features(population)
            evaluated_population = self.with_scores(population)
            evaluated_population = self.truncation_selection(evaluated_population, self.population_size)

            parents = self.select(evaluated_population)
            children = self.get_children(parents)

            archive.update(self.without_scores(parents))
            population = self.without_scores(evaluated_population)  # to add the effect of limit_population_size
            population.extend(children)
            population = self.without_features_in_archive(population, archive)

        winning_features = list(archive)
        evaluated_winners = self.with_scores(winning_features)
        evaluated_winners = self.truncation_selection(evaluated_winners, self.population_size)
        return self.without_scores(evaluated_winners)

    def mine_features_without_archive(self) -> Population:
        population = self.get_initial_population()
        iteration = 0

        def should_continue():
            return (not self.termination_criteria_met(iteration=iteration,
                                                      population=population,
                                                      used_budget=self.feature_selector.used_budget)) and len(
                population) > 0

        while should_continue():
            iteration += 1
            print(f"In iteration {iteration}")
            evaluated_population = self.with_scores(population)
            evaluated_population = self.truncation_selection(evaluated_population, self.population_size)

            parents = self.select(evaluated_population)
            children = self.get_children(parents)

            population = self.without_scores(evaluated_population)  # to add the effect of limit_population_size
            population.extend(children)

        evaluated_winners = self.with_scores(population)
        evaluated_winners = self.truncation_selection(evaluated_winners, self.population_size)
        return self.without_scores(evaluated_winners)


    def mine_features(self) -> list[Feature]:
        if self.uses_archive:
            return self.mine_features_with_archive()
        else:
            return self.mine_features_without_archive()





def run_for_fixed_amount_of_iterations(amount_of_iterations: int) -> Callable:
    def should_terminate(**kwargs):
        return kwargs["iteration"] >= amount_of_iterations

    return should_terminate


def run_for_fixed_budget(budget_limit: int) -> Callable:
    def should_terminate(**kwargs):
        return kwargs["used_budget"] >= budget_limit

    return should_terminate


def found_features(features_to_find: Iterable[Feature], in_archive: bool) -> Callable:
    def should_terminate_archive(**kwargs):
        return all(feature in kwargs["archive"] for feature in features_to_find)

    def should_terminate_population(**kwargs):
        return all(feature in kwargs["population"] for feature in features_to_find)

    if in_archive:
        return should_terminate_archive
    else:
        return should_terminate_population
