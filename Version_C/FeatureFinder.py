import itertools
import math
from typing import Any

import numpy as np
import random

import BenchmarkProblems.CombinatorialProblem
import SearchSpace
import utils
from enum import Enum, auto
from typing import Optional

from Version_B.PopulationSamplePrecomputedData import PopulationSamplePrecomputedData, \
    PopulationSampleWithFeaturesPrecomputedData


from Version_C.Feature import Feature


class ParentPool:
    """this is a data structure to store a list of features and their scores"""
    """ The scores indicate how good a feature is, in terms of explainability and either fitness or novelty,
        These scores go from 0 to 1 and are also used to sample the features using weights.
        The main purpose of this class is to select a random feature and use it as a parent somewhere else
    """
    features: list[Feature]
    weights: list[float]

    precomputed_cumulative_list: list[float]

    def __init__(self, features, weights):
        self.features = features
        self.weights = weights
        self.precomputed_cumulative_list = np.cumsum(weights)

    def select_parent_randomly(self) -> SearchSpace.Feature:
        return random.choices(population=self.features, cum_weights=self.precomputed_cumulative_list)[0]

    def select_n_parents_randomly(self, amount: int):
        return random.choices(population=self.features, cum_weights=self.precomputed_cumulative_list, k=amount)

    def get_raw_features(self) -> list[SearchSpace.Feature]:
        return [feature.to_legacy_feature() for feature in self.features]

    @classmethod
    def get_empty_parent_pool(cls):
        return ParentPool([], [])


class FeatureMixer:
    """ this class takes two sets of parents, and uses them to create new features"""
    """ it simply decides a parent from each set, and combines them"""
    """alternatively, you can use a greedy heuristic approach to get the best n"""
    parent_set_1: ParentPool
    parent_set_2: ParentPool
    """assumes parent_set_2 is either the same as parent_set_1 or bigger in len"""

    asexual: bool

    def __init__(self, parent_set_1: ParentPool, parent_set_2: ParentPool, asexual):
        self.parent_set_1, self.parent_set_2 = parent_set_1, parent_set_2
        if len(parent_set_1.features) > len(parent_set_2.features):
            self.parent_set_1, self.parent_set_2 = self.parent_set_2, self.parent_set_1

        self.asexual = asexual

    def select_parents(self) -> (Feature, Feature):
        return self.parent_set_1.select_parent_randomly(), self.parent_set_2.select_parent_randomly()

    def get_stochastically_mixed_features(self, amount: int) -> list[Feature]:
        batch_size = amount
        result = set()

        def add_from_batch():
            batch_mothers = self.parent_set_1.select_n_parents_randomly(batch_size)
            batch_fathers = self.parent_set_2.select_n_parents_randomly(batch_size)
            offspring = [Feature.merge(mother, father) for mother, father in zip(batch_mothers, batch_fathers)
                         if Feature.are_disjoint(mother, father)]
            result.update(offspring)

        while len(result) < amount:
            add_from_batch()

        return list(result)[:amount]

    @staticmethod
    def add_merged_if_mergeable(accumulator: set[Feature],
                                mother: Feature,
                                father: Feature):
        if Feature.are_disjoint(mother, father):
            child = Feature.merge(mother, father)
            accumulator.add(child)

    def get_heuristic_mixed_features_asexual(self, amount: int):
        result = set()

        for row, row_feature in reversed(list(enumerate(self.parent_set_1.features))):
            if len(result) >= amount:
                break
            for column_feature in self.parent_set_2.features[-1:row:-1]:  # mamma mia
                FeatureMixer.add_merged_if_mergeable(result, row_feature, column_feature)

        return list(result)[:amount]

    def get_heuristic_mixed_features_different_parents(self, amount: int):
        result = set()
        for row_feature in reversed(self.parent_set_1.features):
            if len(result) >= amount:
                break
            for column_feature in reversed(self.parent_set_2.features):
                FeatureMixer.add_merged_if_mergeable(result, row_feature, column_feature)

        return list(result)[:amount]

    def get_heuristically_mixed_features(self, amount: int):
        """this is the greedy heuristic mixing approach"""
        if self.asexual:
            return self.get_heuristic_mixed_features_asexual(amount)
        else:
            return self.get_heuristic_mixed_features_different_parents(amount)


class ScoringCriterion(Enum):
    EXPLAINABILITY = auto()
    HIGH_FITNESS = auto()
    LOW_FITNESS = auto()
    POPULARITY = auto()
    NOVELTY = auto()
    STABILITY = auto()


class FeatureFilter:
    """this class accepts a list of features, and can
        * create scores based on their explainability + fitness / novelty
        * filter the given features based on those scores
    """
    current_features: list[SearchSpace.Feature]
    precomputed_data_for_features: PopulationSampleWithFeaturesPrecomputedData
    complexity_array: np.ndarray

    for_novelty: bool
    expected_proportions: np.ndarray

    def __init__(self,
                 initial_features: list[Feature],
                 precomputed_sample_data,
                 complexity_function,
                 importance_of_explainability,
                 expected_proportions=None):
        self.current_features = initial_features
        stripped_initial_features = [intermediate.to_legacy_feature() for intermediate in initial_features]
        self.precomputed_data_for_features = PopulationSampleWithFeaturesPrecomputedData(precomputed_sample_data,
                                                                                         stripped_initial_features)
        self.complexity_array = np.array([complexity_function(feature) for feature in stripped_initial_features])
        self.importance_of_explainability = importance_of_explainability
        self.for_novelty = (expected_proportions is not None)
        self.expected_proportions = expected_proportions

    def get_complexity_array(self) -> np.ndarray:
        return utils.remap_array_in_zero_one(self.complexity_array)

    def get_explainability_array(self) -> np.ndarray:
        return 1.0 - self.get_complexity_array()

    def get_positive_fitness_correlation_array(self) -> np.ndarray:
        return utils.remap_array_in_zero_one(self.precomputed_data_for_features.get_average_fitness_vector())
        # return utils.remap_array_in_zero_one(self.precomputed_data_for_features.get_t_scores())

    def get_negative_fitness_correlation_array(self) -> np.ndarray:
        return 1.0 - self.get_positive_fitness_correlation_array()

    def get_popularity_array(self):
        observed_proportions = self.precomputed_data_for_features.get_observed_proportions()
        chi_squareds = utils.chi_squared(observed_proportions, self.expected_proportions)
        which_are_good = np.sign(observed_proportions - self.expected_proportions)
        signed_chi_squareds = chi_squareds * which_are_good
        return utils.remap_array_in_zero_one(signed_chi_squareds)

    def get_stability_array(self):
        return utils.remap_array_in_zero_one(self.precomputed_data_for_features.get_t_scores())

    def get_novelty_array(self):
        return 1.0 - self.get_popularity_array()

    def get_requested_score(self, criteria: ScoringCriterion) -> np.ndarray:
        if criteria == ScoringCriterion.EXPLAINABILITY:
            return self.get_explainability_array()
        elif criteria == ScoringCriterion.HIGH_FITNESS:
            return self.get_positive_fitness_correlation_array()
        elif criteria == ScoringCriterion.LOW_FITNESS:
            return self.get_negative_fitness_correlation_array()
        elif criteria == ScoringCriterion.POPULARITY:
            return self.get_popularity_array()
        elif criteria == ScoringCriterion.NOVELTY:
            return self.get_novelty_array()
        elif criteria == ScoringCriterion.STABILITY:
            return self.get_stability_array()
        else:
            raise Exception("The criterion for scoring was not specified")

    def get_scores_of_features(self, additional_criteria: Optional[ScoringCriterion]) -> np.ndarray:
        explainabilities = self.get_requested_score(ScoringCriterion.EXPLAINABILITY)

        if additional_criteria is None:
            return explainabilities

        criteria_scores = self.get_requested_score(additional_criteria)
        return utils.weighted_sum(explainabilities, self.importance_of_explainability,
                                  criteria_scores, 1.0 - self.importance_of_explainability)

    def get_the_best_features(self, how_many_to_keep: int,
                              additional_criteria: Optional[ScoringCriterion]) -> (
            list[Feature], np.ndarray):
        scores = self.get_scores_of_features(additional_criteria)

        sorted_by_with_score = sorted(zip(self.current_features, scores), key=utils.second, reverse=True)
        features, scores_list = utils.unzip(sorted_by_with_score[:how_many_to_keep])
        return features, np.array(scores_list)


class FeatureDeveloper:
    """this class generates the useful, explainable features"""
    population_sample: PopulationSamplePrecomputedData
    previous_iterations: list[ParentPool]
    guaranteed_depth: int
    search_space: SearchSpace.SearchSpace
    complexity_function: Any  # SearchSpace.Feature -> float
    importance_of_explainability: float
    additional_criteria: Optional[ScoringCriterion]
    amount_requested: int
    thoroughness: float

    def get_filter(self, intermediates: list[Feature]):
        expected_proportions = None
        if self.additional_criteria == ScoringCriterion.NOVELTY:
            expected_proportions = np.array([self.search_space.probability_of_feature_in_uniform(intermediate.to_legacy_feature())
                                             for intermediate in intermediates])

        return FeatureFilter(intermediates,
                             self.population_sample,
                             self.complexity_function,
                             self.importance_of_explainability,
                             expected_proportions=expected_proportions)

    def get_trivial_parent_pool(self):
        trivial_features = Feature.get_all_trivial_features(self.search_space)

        feature_filter = self.get_filter(trivial_features)
        scores = feature_filter.get_scores_of_features(additional_criteria=None)

        return ParentPool(trivial_features, scores)  # note how they don't get filtered!

    def __init__(self,
                 search_space,
                 population_sample,
                 guaranteed_depth,
                 extra_depth,
                 complexity_function,
                 importance_of_explainability,
                 additional_criteria: Optional[ScoringCriterion],
                 amount_requested):
        self.search_space = search_space
        self.population_sample = population_sample
        self.guaranteed_depth = guaranteed_depth
        self.extra_depth = extra_depth
        self.complexity_function = complexity_function
        self.importance_of_explainability = importance_of_explainability
        self.additional_criteria = additional_criteria
        self.previous_iterations = [self.get_trivial_parent_pool()]
        self.amount_requested = amount_requested

    def get_parent_pool_of_weight(self, weight):
        return self.previous_iterations[weight - 1]

    def get_mixing_parent_weights(self, weight):
        parent_weight_1 = weight // 2
        parent_weight_2 = weight - parent_weight_1
        return parent_weight_1, parent_weight_2

    def get_amount_of_features_of_weight(self, weight):
        return len(self.get_parent_pool_of_weight(weight).features)

    def new_iteration(self,
                      amount_to_consider: int,
                      amount_to_return: int,
                      heuristic=False,
                      use_additional_criteria=True):
        new_weight = len(self.previous_iterations) + 1
        parent_weight_1, parent_weight_2 = self.get_mixing_parent_weights(new_weight)

        asexual_mixing = parent_weight_1 == parent_weight_2
        feature_mixer = FeatureMixer(self.get_parent_pool_of_weight(parent_weight_1),
                                     self.get_parent_pool_of_weight(parent_weight_2),
                                     asexual_mixing)

        if heuristic:
            considered_features = feature_mixer.get_heuristically_mixed_features(amount_to_consider)
        else:
            considered_features = feature_mixer.get_stochastically_mixed_features(amount_to_consider)

        if len(considered_features) == 0:
            self.previous_iterations.append(ParentPool.get_empty_parent_pool())
            return
        feature_filter = self.get_filter(considered_features)

        features, scores = feature_filter.get_the_best_features(amount_to_return,
                                                                additional_criteria=self.additional_criteria
                                                                if use_additional_criteria else None)
        self.previous_iterations.append(ParentPool(features, scores))

    def get_schedule(self, heuristic_strategy="where needed") -> list[(int, bool, bool, int, int)]:
        def get_total_amount_of_features(weight_category: int):
            """if all the variables had the same cardinality, this would be simpler"""
            return sum([utils.product(which_vars) for which_vars in
                        itertools.combinations(self.search_space.cardinalities, weight_category)])


        def amount_to_keep_for_weight(weight_category: int):
            if weight_category <= self.guaranteed_depth:
                return get_total_amount_of_features(weight_category)
            else:
                return self.amount_requested


        def amount_to_consider_for_weight(weight_category: int, kept_in_category: int):
            if weight_category <= self.guaranteed_depth:
                return kept_in_category
            else:
                return kept_in_category * 2 # TODO think about this more

        def should_use_heuristic(weight_category: int):
            return weight_category <= self.guaranteed_depth


        def should_use_criteria(weight_category: int):
            return weight_category > (self.guaranteed_depth-1)


        weights = list(range(2, self.guaranteed_depth+1))
        should_be_heuristic = [should_use_heuristic(weight) for weight in weights]
        should_use_criteria = [should_use_criteria(weight) for weight in weights]
        kepts = [amount_to_keep_for_weight(weight) for weight in weights]
        considereds = [amount_to_consider_for_weight(weight, kept) for weight, kept in zip(weights, kepts)]

        return zip(weights, should_be_heuristic, should_use_criteria, kepts, considereds)


    def develop_features(self, heuristic_strategy):

        settings_schedule = self.get_schedule(heuristic_strategy)

        for weight, use_heuristic, use_criteria, amount_to_keep, amount_to_consider in settings_schedule:
            print(f"In iteration where {weight = }: {use_heuristic = }, {amount_to_keep = }, {amount_to_consider}")
            self.new_iteration(amount_to_consider,
                               amount_to_keep,
                               heuristic=use_heuristic,
                               use_additional_criteria=use_criteria)

    def get_developed_features(self) -> (list[SearchSpace.Feature], np.ndarray):
        """This is the function which returns the features you'll be using in the future!"""
        developed_features: list[Feature] = utils.concat([parent_pool.features
                                                                      for parent_pool in self.previous_iterations])
        feature_filterer = self.get_filter(developed_features)

        return feature_filterer.get_the_best_features(self.amount_requested, self.additional_criteria)


def find_features(problem: BenchmarkProblems.CombinatorialProblem.CombinatorialProblem,
                  guaranteed_depth: int,
                  extra_depth: int,
                  importance_of_explainability: float,
                  sample_data: PopulationSamplePrecomputedData,
                  heuristic_strategy: Any,
                  criteria: Optional[ScoringCriterion],
                  amount_requested: int) -> (list[SearchSpace.Feature], np.ndarray):
    feature_developer = FeatureDeveloper(search_space=problem.search_space,
                                         population_sample=sample_data,
                                         guaranteed_depth=guaranteed_depth,
                                         extra_depth=extra_depth,
                                         complexity_function=problem.get_complexity_of_feature,
                                         importance_of_explainability=importance_of_explainability,
                                         additional_criteria=criteria,
                                         amount_requested=amount_requested)

    feature_developer.develop_features(heuristic_strategy=heuristic_strategy)

    intermediate_features, scores = feature_developer.get_developed_features()
    raw_features = [intermediate.to_legacy_feature() for intermediate in intermediate_features]

    # give_explainability_and_average_fitnesses(raw_features, sample_data, problem, criteria)
    return raw_features, scores


def give_explainability_and_average_fitnesses(features: list[SearchSpace.Feature],
                                              population_sample: PopulationSamplePrecomputedData,
                                              problem: BenchmarkProblems.CombinatorialProblem.CombinatorialProblem,
                                              criteria: Optional[ScoringCriterion]):
    complexities = np.array([problem.get_complexity_of_feature(feature) for feature in features])

    sample_with_features = PopulationSampleWithFeaturesPrecomputedData(population_sample, features)
    criteria_scores = None
    if criteria in {ScoringCriterion.LOW_FITNESS, ScoringCriterion.HIGH_FITNESS}:
        criteria_scores = sample_with_features.get_average_fitness_vector()
    elif criteria in {ScoringCriterion.NOVELTY, ScoringCriterion.POPULARITY}:
        criteria_scores = sample_with_features.get_observed_proportions()
    else:
        criteria_scores = np.zeros_like(complexities)

    for feature, score_for_criteria, complexity in zip(features, criteria_scores, complexities):
        print(f"{problem.feature_repr(feature)}\nHas {score_for_criteria = :.3f}, {complexity = :.2f}")

    # TODO:

    """ 
        * Clean up VariateModels
        
        * create the class which stores iterations and selects from them 
            * perhaps FeatureFinder, which makes use of featurefilter to obtain scores and decide what to keep?
            * perhaps there's 3 phases to featurefilter (3 states):
                * struggle: be fed the features obtained from merging previous iterations
                * grind: give scores to the features, select the best ones
                * shine: remove the unwanted features, we store only the good features and their scores
                
                
        TODO: currently if you set expected_proportions to none, it means you only look at the fitness
        if it's set, it will only look at the novelty. This is confusing and nasty.
        
        * decide how the greedy heuristic approach should consider more features before returning.
    """
