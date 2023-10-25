import math
import random
import time
from typing import Iterable

import numpy as np

import utils
from Version_E import HotEncoding
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, FeatureSelector
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation

Score = float
class ArchiveMiner(FeatureMiner):
    population_size: int
    generations: int

    Population = list[Feature]
    EvaluatedPopulation = list[(Feature, float)]

    def __init__(self, selector: FeatureSelector, population_size: int, generations: int):
        super().__init__(selector)
        self.population_size = population_size
        self.generations = generations

    def __repr__(self):
        return (f"ArchiveMiner(population = {self.population_size}, "
                f"generations = {self.generations})")

    def with_scores(self, feature_list: Population) -> EvaluatedPopulation:
        scores = self.feature_selector.get_scores(feature_list)
        return list(zip(feature_list, scores))

    def without_scores(self, feature_list: EvaluatedPopulation) -> Population:
        return utils.unzip(feature_list)[0]

    def remove_duplicate_features(self, features: Population) -> Population:
        return list(set(features))

    def truncation_selection(self, evaluated_features: EvaluatedPopulation,
                             how_many_to_keep: int) -> EvaluatedPopulation:
        evaluated_features.sort(key=utils.second, reverse=True)
        return evaluated_features[:how_many_to_keep]

    def tournament_selection(self, evaluated_features: EvaluatedPopulation,
                             how_many_to_keep: int) -> EvaluatedPopulation:
        tournament_size = 12

        scores = utils.unzip(evaluated_features)[1]
        cumulative_probabilities = np.cumsum(scores)

        def get_tournament_pool() -> list[(Feature, float)]:
            return random.choices(evaluated_features, cum_weights=cumulative_probabilities, k=tournament_size)

        def pick_winner(tournament_pool: list[Feature, float]) -> (Feature, float):
            return max(tournament_pool, key=utils.second)

        return list(utils.generate_distinct(lambda: pick_winner(get_tournament_pool()), how_many_to_keep))

    def fitness_proportionate_selection(self, evaluated_features: EvaluatedPopulation,
                                        how_many_to_keep: int) -> EvaluatedPopulation:
        batch_size = 256
        scores = utils.unzip(evaluated_features)[1]
        cumulative_probabilities = np.cumsum(scores)
        def get_batch():
            return random.choices(evaluated_features, cum_weights=cumulative_probabilities, k=batch_size)

        accumulator = set()
        while len(accumulator) < how_many_to_keep:
            accumulator.update(get_batch())

        return list(accumulator)[:how_many_to_keep]

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
        return [child for parent, score in parents for child in Feature.get_specialisations(parent, self.search_space)]

    def get_initial_population(self) -> Population:
        raise Exception("An implementation of ArchiveMiner does not implement get_initial_population")

    def get_children(self, parents: EvaluatedPopulation) -> list[Feature]:
        raise Exception("An implementation of ArchiveMiner does not implement get_children")

    def select(self, population: EvaluatedPopulation) -> EvaluatedPopulation:
        raise Exception("An implementation of ArchiveMiner does not implement select")

    def mine_features(self) -> Population:
        population = self.get_initial_population()
        archive = set()

        for iteration in range(self.generations):
            if len(population) == 0:
                break
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
