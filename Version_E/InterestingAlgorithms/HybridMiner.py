import math
import random
from enum import auto, Enum

import numpy as np

import utils
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import LayeredFeatureMiner, FeatureSelector, FeatureMiner
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation


class HybridMiner(FeatureMiner):
    population_size: int
    stochastic: bool
    generations: int

    def __init__(self, selector: FeatureSelector, population_size: int, generations: int, stochastic: bool):
        super().__init__(selector)
        self.population_size = population_size
        self.generations = generations
        self.stochastic = stochastic

    def __repr__(self):
        return (f"Hybrid(population = {self.population_size}, "
                f"generations = {self.generations},"
                f"stochastic = {self.stochastic})")

    def select_parents(self, current_population: list[(Feature, float)]) -> list[Feature]:
        portion_to_select = 0.5
        amount_to_select = math.ceil(len(current_population) * portion_to_select)
        if self.stochastic:
            return self.feature_selector.select_features_stochastically(current_population, amount_to_select)
        else:
            return self.feature_selector.select_features_heuristically(current_population, amount_to_select)

    def get_next_population(self, prev_population: list[(Feature, float)]) -> list[(Feature, float)]:
        selected_parents = self.select_parents(prev_population)
        simplifications = {simplified for feature in selected_parents
                           for simplified in feature.get_generalisations()}

        complications = {complected for feature in selected_parents
                         for complected in feature.get_specialisations(self.search_space)}

        prev_population_as_set = {feature for feature, score in prev_population}
        unselected_new_population = simplifications.union(complications).union(prev_population_as_set)
        return self.feature_selector.keep_best_features(unselected_new_population, self.population_size)


    def get_initial_population(self) -> list[(Feature, float)]:
        initial_features = [Feature.empty_feature(self.search_space)]
        scores = self.feature_selector.get_scores(initial_features)
        return list(zip(initial_features, scores))

    def get_max_score_of_population(self, population: list[(Feature, float)]) -> float:
        return max(population, key = utils.second)[1]

    def mine_features(self) -> list[Feature]:
        current_population = self.get_initial_population()

        for iteration in range(self.generations):
            print(f"Processing iteration {iteration} / {self.generations}")
            current_population = self.get_next_population(current_population)

        return utils.unzip(current_population)[0]


class ArchiveMiner(FeatureMiner):
    population_size: int
    stochastic: bool
    generations: int

    Population = list[Feature]
    EvaluatedPopulation = list[(Feature, float)]

    def __init__(self, selector: FeatureSelector, population_size: int, generations: int, stochastic: bool):
        super().__init__(selector)
        self.population_size = population_size
        self.generations = generations
        self.stochastic = stochastic

    def __repr__(self):
        return (f"ArchiveMiner(population = {self.population_size}, "
                f"generations = {self.generations},"
                f"stochastic = {self.stochastic})")


    def get_children(self, feature: Feature) -> list[Feature]:
        result = feature.get_specialisations(self.search_space)
        result.extend(feature.get_generalisations())
        return result


    def with_scores(self, feature_list: Population) -> EvaluatedPopulation:
        scores = self.feature_selector.get_scores(feature_list)
        return list(zip(feature_list, scores))

    def without_scores(self, feature_list: EvaluatedPopulation) -> Population:
        return utils.unzip(feature_list)[0]

    def remove_duplicate_features(self, features: Population) -> Population:
        return list(set(features))

    def truncation_selection(self, evaluated_features: EvaluatedPopulation, how_many_to_keep: int) -> EvaluatedPopulation:
        evaluated_features.sort(key=utils.second, reverse=True)
        return evaluated_features[:how_many_to_keep]


    def tournament_selection(self, evaluated_features: EvaluatedPopulation, how_many_to_keep: int) -> EvaluatedPopulation:
        tournament_size = 12

        scores = utils.unzip(evaluated_features)[1]
        cumulative_probabilities = np.cumsum(scores)
        def get_tournament_pool() -> list[(Feature, float)]:
            return random.choices(evaluated_features, cum_weights=cumulative_probabilities, k=tournament_size)

        def pick_winner(tournament_pool: list[Feature, float]) -> (Feature, float):
            return max(tournament_pool, key=utils.second)

        return list(utils.generate_distinct(lambda: pick_winner(get_tournament_pool()), how_many_to_keep))

    def limit_population_size(self, evaluated_features: EvaluatedPopulation) -> EvaluatedPopulation:
        return self.truncation_selection(evaluated_features, self.population_size)



    def select_parents(self, evaluated_population: EvaluatedPopulation) -> Population:
        parents_proportion = 0.3
        amount_of_parents = math.ceil(len(evaluated_population)*parents_proportion)
        if self.stochastic:
            selected_with_scores = self.tournament_selection(evaluated_population, amount_of_parents)
        else:
            selected_with_scores = self.truncation_selection(evaluated_population, amount_of_parents)

        return self.without_scores(selected_with_scores)

    def without_features_in_archive(self, population: Population, archive: set[Feature]) -> Population:
        return [feature for feature in population if feature not in archive]



    def mine_features_old(self) -> Population:
        population = [Feature.empty_feature(self.search_space)]
        archive = set()

        for iteration in range(self.generations):
            print(f"In iteration {iteration}")
            evaluated_population = self.with_scores(population)
            parents = self.select_parents(evaluated_population)
            children = [child for parent in parents for child in self.get_children(parent)]

            archive.update(parents)
            population.extend(children)
            population = self.without_features_in_archive(population, archive)

            population = self.limit_population_size(population)


        winning_features = list(archive)
        evaluated_winners = self.with_scores(winning_features)
        evaluated_winners = self.truncation_selection(evaluated_winners, self.population_size)
        return self.without_scores(evaluated_winners)


    def mine_features(self) -> Population:
        population = [Feature.empty_feature(self.search_space)]
        archive = set()

        for iteration in range(self.generations):
            print(f"In iteration {iteration}")
            population = self.remove_duplicate_features(population)
            evaluated_population = self.with_scores(population)
            evaluated_population = self.limit_population_size(evaluated_population)

            parents = self.select_parents(evaluated_population)
            children = [child for parent in parents for child in self.get_children(parent)]

            archive.update(parents)
            population = self.without_scores(evaluated_population)  # to add the effect of limit_population_size
            population.extend(children)
            population = self.without_features_in_archive(population, archive)

        winning_features = list(archive)
        evaluated_winners = self.with_scores(winning_features)
        evaluated_winners = self.truncation_selection(evaluated_winners, self.population_size)
        return self.without_scores(evaluated_winners)


