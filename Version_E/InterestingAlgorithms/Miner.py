import math
import random
import time
from typing import Iterable, Callable

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

    used_budget: int

    def __init__(self, ppi: PrecomputedPopulationInformation, criterion: MeasurableCriterion):
        self.ppi = ppi
        self.criterion = criterion
        self.used_budget = 0

    def __repr__(self):
        return "FeatureSelector"

    def get_scores(self, features: list[Feature]) -> np.ndarray:
        if len(features) == 0:
            raise Exception("Not enough features to give a relative score")
        self.used_budget += len(features)
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
    termination_criteria_met: Callable

    Population = list[Feature]
    EvaluatedPopulation = list[(Feature, float)]

    def __init__(self, feature_selector: FeatureSelector, termination_criteria_met: Callable):
        self.feature_selector = feature_selector
        self.termination_criteria_met = termination_criteria_met

    @property
    def search_space(self):
        return self.feature_selector.ppi.search_space

    def mine_features(self) -> list[Feature]:
        raise Exception("An implementation of FeatureMiner does not implement mine_features")

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

        # return list(utils.generate_distinct(lambda: pick_winner(get_tournament_pool()), how_many_to_keep))   if you want them distinct
        return [pick_winner(get_tournament_pool()) for _ in range(how_many_to_keep)]

    def fitness_proportionate_selection(self, evaluated_features: EvaluatedPopulation,
                                        how_many_to_keep: int) -> EvaluatedPopulation:
        batch_size = 256
        scores = utils.unzip(evaluated_features)[1]
        cumulative_probabilities = np.cumsum(scores)

        # def get_batch():
        #     return random.choices(evaluated_features, cum_weights=cumulative_probabilities, k=batch_size)
        #
        # accumulator = set()
        # while len(accumulator) < how_many_to_keep:
        #     accumulator.update(get_batch())

        # return list(accumulator)[:how_many_to_keep]    if you want them distinct

        return random.choices(evaluated_features, cum_weights=cumulative_probabilities, k=how_many_to_keep)

    def get_meaningful_features(self, amount_to_return: int) -> list[Feature]:
        mined_features = self.mine_features()

        kept_features_with_scores = sorted(self.with_scores(mined_features), key=utils.second, reverse=True)
        kept_features_with_scores = kept_features_with_scores[:amount_to_return]
        return [feature for feature, score in kept_features_with_scores]




def run_for_fixed_amount_of_iterations(amount_of_iterations: int) -> Callable:
    def should_terminate(**kwargs):
        return kwargs["iteration"] >= amount_of_iterations

    return should_terminate


def run_with_limited_budget(budget_limit: int) -> Callable:
    def should_terminate(**kwargs):
        return kwargs["used_budget"] >= budget_limit

    return should_terminate


def run_until_found_features(features_to_find: Iterable[Feature], max_budget: int) -> Callable:

    def all_are_found_in(collection: Iterable[Feature]):
        return all(feature in collection for feature in features_to_find)
    def should_terminate(**kwargs):
        if (kwargs["used_budget"] >= max_budget):
            return True
        return all_are_found_in(kwargs["archive"]) or all_are_found_in(kwargs["population"])

    return should_terminate


