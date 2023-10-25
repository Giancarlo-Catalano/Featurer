import math
import random
import time
from typing import Iterable

import numpy as np

import utils
from Version_E import HotEncoding
from Version_E.Feature import Feature
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation

Score = float


class FeatureSelector:
    ppi: PrecomputedPopulationInformation
    criterion: MeasurableCriterion

    def __init__(self, ppi: PrecomputedPopulationInformation, criterion: MeasurableCriterion):
        self.ppi = ppi
        self.criterion = criterion

    def __repr__(self):
        return "FeatureSelector"

    def get_scores(self, features: list[Feature]) -> np.ndarray:
        pfi = PrecomputedFeatureInformation(self.ppi, features)
        return self.criterion.get_score_array(pfi)

    def keep_best_features(self, features: Iterable[Feature], amount_to_keep: int) -> list[(Feature, Score)]:
        features_list = list(features)
        scores = self.get_scores(features_list)
        sorted_paired = sorted(zip(features_list, scores), key=utils.second, reverse=True)
        return sorted_paired[:amount_to_keep]

    def select_features_stochastically(self, features_with_scores: list[(Feature, Score)],
                                       amount_to_return: int) -> list[Feature]:
        """
        This function selects features using tournament selection, returning a distinct set of features
        :param features_with_scores: layer to select from
        :param amount_to_return: Res Ipsa Loquitur
        :return:
        """
        features, weights = utils.unzip(features_with_scores)
        random.seed(int(time.time()))  # to prevent predictable randomness
        selected = random.choices(features, weights=weights, k=amount_to_return)

        return list(selected)

    def select_features_heuristically(self, features_with_scores: list[(Feature, Score)], amount_to_return: int) -> \
    list[Feature]:
        """Select features using truncation selection"""
        # ensure previous layer is sorted
        features_with_scores.sort(key=utils.second, reverse=True)
        return [feature for feature, _ in features_with_scores[:amount_to_return]]


class FeatureMiner:
    feature_selector: FeatureSelector

    def __init__(self, feature_selector: FeatureSelector):
        self.feature_selector = feature_selector

    @property
    def search_space(self):
        return self.feature_selector.ppi.search_space

    def mine_features(self) -> list[Feature]:
        raise Exception("An implementation of FeatureMiner does not implement mine_features")

    def cull_subsets(self, features: list[Feature]) -> list[(Feature, Score)]:
        # TODO this is not working as intended, some subsets are still present at the end!!!
        features_with_scores = list(zip(features, self.feature_selector.get_scores(features)))
        features_with_scores.sort(key=utils.second, reverse=True)
        kept = []

        def consider_feature(feature: Feature, score: Score):
            for index, (other_feature, other_score) in enumerate(kept):
                if feature.is_subset_of(other_feature) or other_feature.is_subset_of(feature):
                    if score > other_score:
                        kept[index] = feature, score
                    return
            kept.append((feature, score))

        for feature, score in features_with_scores:
            consider_feature(feature, score)

        return kept

    def get_meaningful_features(self, amount_to_return: int, cull_subsets=False) -> list[Feature]:
        mined_features = self.mine_features()
        if cull_subsets:
            culled_features = self.cull_subsets(mined_features)
        else:
            culled_features = zip(mined_features, self.feature_selector.get_scores(mined_features))

        kept_features_with_scores = sorted(culled_features, key=utils.second, reverse=True)[:amount_to_return]
        return [feature
                for feature, score in kept_features_with_scores]


Layer = list[(Feature, Score)]


class LayeredFeatureMiner(FeatureMiner):
    amount_to_keep_in_each_layer: int
    stochastic: bool

    def __init__(self, selector: FeatureSelector, amount_to_keep_in_each_layer: int, stochastic: bool):
        super().__init__(selector)
        self.amount_to_keep_in_each_layer = amount_to_keep_in_each_layer
        self.stochastic = stochastic

    def get_initial_features(self, ppi: PrecomputedPopulationInformation) -> list[Feature]:
        raise Exception("An implementation of FeatureMiner does not implement get_initial_layer")

    def modifications_of_feature(self, feature: Feature) -> list[Feature]:
        raise Exception("An implementation of FeatureMiner does not implement branch_from_feature")

    def should_terminate(self, next_iteration: int):
        raise Exception("An implementation of FeatureMixer does not implement should_terminate")

    def make_layer(self, features: list[Feature]) -> Layer:
        return list(zip(features, self.feature_selector.get_scores(features)))

    def keep_top_features_and_make_layer(self, features: list[Feature], amount_to_keep: int) -> Layer:
        return self.feature_selector.keep_best_features(features, amount_to_keep)

    def select_features(self, previous_layer) -> list[Feature]:
        """
        This function selects a portion of the population, using a different method based on internal parameters
        :param previous_layer: the layer to select from
        :return: returns a set of features
        """

        proportion = 0.5
        amount_to_return = math.ceil(len(previous_layer) * proportion)
        if len(previous_layer) <= amount_to_return:
            return utils.unzip(previous_layer)[0]

        if self.stochastic:
            return self.feature_selector.select_features_stochastically(previous_layer, amount_to_return)
        else:
            return self.feature_selector.select_features_heuristically(previous_layer, amount_to_return)

    def mine_features(self) -> list[Feature]:

        def truncate_and_make_layer(features: list[Feature]) -> Layer:
            return self.feature_selector.keep_best_features(features, self.amount_to_keep_in_each_layer)

        def get_initial_layer():
            initial_features = self.get_initial_features(self.feature_selector.ppi)
            return truncate_and_make_layer(initial_features)

        def get_next_layer(prev_layer: Layer) -> Layer:
            selected_features: list[Feature] = self.select_features(prev_layer)

            modified_features = [modified_feature
                                 for feature in selected_features
                                 for modified_feature in self.modifications_of_feature(feature)]
            modified_features = utils.remove_duplicates(modified_features, hashable=True)
            return truncate_and_make_layer(modified_features)

        layers: list[Layer] = [get_initial_layer()]

        iteration = 0
        while True:
            iteration += 1
            # print(f"We're at iteration {iteration}")
            if self.should_terminate(iteration):
                break
            layers.append(get_next_layer(layers[-1]))

        final_features = [feature
                          for layer in layers
                          for feature, _ in layer]
        return final_features


class ArchiveMiner(FeatureMiner):
    population_size: int
    generations: int

    Population = list[Feature]
    EvaluatedPopulation = list[(Feature, float)]

    def __init__(self, selector: FeatureSelector, population_size: int, generations: int, stochastic: bool):
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

    def get_complected_children(self, parents: Population) -> list[Feature]:
        return [child for parent in parents for child in Feature.get_specialisations(parent, self.search_space)]

    def get_simplified_children(self, parents: Population) -> list[Feature]:
        return [child for parent in parents for child in Feature.get_specialisations(parent, self.search_space)]

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
            print(f"In iteration {iteration}")
            population = self.remove_duplicate_features(population)
            evaluated_population = self.with_scores(population)
            evaluated_population = self.truncation_selection(evaluated_population, self.population_size)

            parents = self.select(evaluated_population)
            children = [child for parent in parents for child in self.get_children(parent)]

            archive.update(parents)
            population = self.without_scores(evaluated_population)  # to add the effect of limit_population_size
            population.extend(children)
            population = self.without_features_in_archive(population, archive)

        winning_features = list(archive)
        evaluated_winners = self.with_scores(winning_features)
        evaluated_winners = self.truncation_selection(evaluated_winners, self.population_size)
        return self.without_scores(evaluated_winners)
