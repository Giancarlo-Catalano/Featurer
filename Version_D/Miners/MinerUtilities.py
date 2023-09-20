import random
from typing import Iterable

import utils
from Version_D.Feature import Feature
from Version_D.MeasurableCriterion import LayerScoringCriteria, compute_scores_for_features
from Version_D.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_D.PrecomputedPopulationInformation import PrecomputedPopulationInformation

Score = float
Layer = dict[Feature, Score]


class FeatureSelector:
    ppi: PrecomputedPopulationInformation
    criteria_and_weights: LayerScoringCriteria

    def __init__(self, ppi: PrecomputedPopulationInformation, criteria_and_weights: LayerScoringCriteria):
        self.ppi = ppi
        self.criteria_and_weights = criteria_and_weights

    def make_layer(self, features: list[Feature]) -> Layer:
        pfi = PrecomputedFeatureInformation(self.ppi, features)
        scores = compute_scores_for_features(pfi, self.criteria_and_weights)
        return dict(zip(features, scores))

    def keep_top_features_and_make_layer(self, features: list[Feature], amount_to_keep: int) -> Layer:
        pfi = PrecomputedFeatureInformation(self.ppi, features)
        scores = compute_scores_for_features(pfi, self.criteria_and_weights)
        sorted_paired = sorted(zip(features, scores), key=utils.second, reverse=True)
        return dict(sorted_paired[:amount_to_keep])


class FeatureMiner:
    feature_selector: FeatureSelector
    amount_to_keep_in_each_layer: int
    stochastic: bool

    def __init__(self, selector: FeatureSelector, amount_to_keep_in_each_layer: int, stochastic: bool):
        self.feature_selector = selector
        self.amount_to_keep_in_each_layer = amount_to_keep_in_each_layer
        self.stochastic = stochastic

    def get_initial_features(self, ppi: PrecomputedPopulationInformation) -> list[Feature]:
        raise Exception("An implementation of FeatureMiner does not implement get_initial_layer")

    def branch_from_feature(self, feature: Feature) -> set[Feature]:
        raise Exception("An implementation of FeatureMiner does not implement branch_from_feature")

    def should_terminate(self, next_iteration: int):
        raise Exception("An implementation of FeatureMixer does not implement should_terminate")

    @property
    def search_space(self):
        return self.feature_selector.ppi.search_space

    def choose_features_stochastically(self, previous_layer: Layer) -> set[Feature]:
        tournament_size = 30
        features_with_scores = list(previous_layer.items())
        weights = utils.unzip(features_with_scores)[1]

        def tournament_select() -> Feature:
            extracted = random.choices(features_with_scores, weights, k=tournament_size)
            return max(extracted, key=utils.second)[0]

        selected = set()
        while len(selected) < self.amount_to_keep_in_each_layer:
            selected.add(tournament_select())

        return selected

    def select_features(self, previous_layer):
        if len(previous_layer) <= self.amount_to_keep_in_each_layer:
            return set(previous_layer.keys())

        if self.stochastic:
            return self.choose_features_stochastically(previous_layer)
        else:
            return set(previous_layer.keys())

    def make_layer_by_truncation(self, features: list[Feature]):
        return self.feature_selector.keep_top_features_and_make_layer(features, self.amount_to_keep_in_each_layer)

    def get_next_layer(self, prev_layer: Layer) -> Layer:
        selected_features = self.select_features(prev_layer)

        modified_features = utils.concat_sets(self.branch_from_feature(feature) for feature in selected_features)
        return self.make_layer_by_truncation(modified_features)

    def mine_features(self) -> list[Feature]:
        initial_features = self.get_initial_features(self.feature_selector.ppi)
        initial_layer = self.feature_selector.make_layer(initial_features)
        layers: list[dict[Feature, float]] = [initial_layer]

        iteration = 0
        while True:
            iteration += 1
            print(f"We're at iteration {iteration}")
            if self.should_terminate(iteration):
                break

            layers.append(self.get_next_layer(layers[-1]))

        final_features = utils.concat_lists(list(layer.keys()) for layer in layers)
        overall_layer = self.make_layer_by_truncation(final_features)

        return list(overall_layer.keys())


class ConstructiveMiner(FeatureMiner):
    at_most_parameters: int

    def __init__(self, selector: FeatureSelector, amount_to_keep_in_each_layer: int, stochastic: bool,
                 at_most_parameters: int):
        super().__init__(selector, amount_to_keep_in_each_layer, stochastic)
        self.at_most_parameters = at_most_parameters

    def get_initial_features(self, ppi: PrecomputedPopulationInformation) -> list[Feature]:
        return [Feature.empty_feature(self.search_space)]

    def branch_from_feature(self, feature: Feature) -> set[Feature]:
        return feature.get_specialisations(self.search_space)

    def should_terminate(self, next_iteration: int):
        next_amount_of_parameters = next_iteration
        return next_iteration > self.at_most_parameters


class DestructiveMiner(FeatureMiner):
    at_least_parameters: int

    def __init__(self, selector: FeatureSelector, amount_to_keep_in_each_layer: int, stochastic: bool,
                 at_least_parameters: int):
        super().__init__(selector, amount_to_keep_in_each_layer, stochastic)
        self.at_least_parameters = at_least_parameters

    def get_initial_features(self, ppi: PrecomputedPopulationInformation) -> list[Feature]:
        return Feature.candidate_matrix_to_features(ppi.candidate_matrix, self.search_space)

    def branch_from_feature(self, feature: Feature) -> set[Feature]:
        return feature.get_generalisations()

    def should_terminate(self, next_iteration: int):
        next_amount_of_parameters = self.search_space.dimensions - next_iteration
        return next_amount_of_parameters < self.at_least_parameters
