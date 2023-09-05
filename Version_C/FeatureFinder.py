import itertools
from typing import Any

import numpy as np
import random

import BenchmarkProblems.CombinatorialProblem
import SearchSpace
import utils
from enum import Enum, auto

from Version_C.PopulationSamplePrecomputedData import PopulationSamplePrecomputedData, \
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
    strict: bool  # determines whether we allow overlapping features to be merged or not
    parent_set_1: ParentPool
    parent_set_2: ParentPool
    """assumes parent_set_2 is either the same as parent_set_1 or bigger in len"""

    asexual: bool

    def __init__(self, parent_set_1: ParentPool, parent_set_2: ParentPool, asexual):
        self.strict = True
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
            for mother, father in zip(batch_mothers, batch_fathers):
                self.add_merged_if_mergeable(result, mother, father)

        while len(result) < amount:
            add_from_batch()

        return list(result)[:amount]

    def add_merged_if_mergeable(self, accumulator: set[Feature],
                                mother: Feature,
                                father: Feature):
        if self.strict and Feature.are_disjoint(mother, father):
            child = Feature.merge(mother, father)
            accumulator.add(child)

    def get_heuristic_mixed_features_asexual(self, amount: int):
        result = set()

        for row, row_feature in reversed(list(enumerate(self.parent_set_1.features))):
            if len(result) >= amount:
                break
            for column_feature in self.parent_set_2.features[-1:row:-1]:  # mamma mia
                self.add_merged_if_mergeable(result, row_feature, column_feature)

        return list(result)[:amount]

    def get_heuristic_mixed_features_different_parents(self, amount: int):
        result = set()
        for row_feature in reversed(self.parent_set_1.features):
            if len(result) >= amount:
                break
            for column_feature in reversed(self.parent_set_2.features):
                self.add_merged_if_mergeable(result, row_feature, column_feature)

        return list(result)[:amount]

    def get_heuristically_mixed_features(self, amount: int):
        """this is the greedy heuristic mixing approach"""
        if self.asexual:
            return self.get_heuristic_mixed_features_asexual(amount)
        else:
            return self.get_heuristic_mixed_features_different_parents(amount)

    def disable_strict_mixing(self):
        self.strict = False


class ScoringCriterion(Enum):
    EXPLAINABILITY = auto()
    HIGH_FITNESS = auto()
    LOW_FITNESS = auto()
    FITNESS_CONSISTENCY = auto()
    POPULARITY = auto()
    NOVELTY = auto()
    RESILIENCY = auto()
    INSTABILITY = auto()
    CORRELATION = auto()
    ANTICORRELATION = auto()

    def __repr__(self):
        return ["Explainability", "High Fitness", "Low Fitness", "Fitness Consistency", "Popularity", "Novelty",
                "Resiliency", "Instability", "Correlation", "Anti Correlation"][self.value - 1]

    def __str__(self):
        return self.__repr__()


class FeatureFilter:
    """this class accepts a list of features, and can
        * create scores based on their explainability + fitness / novelty
        * filter the given features based on those scores
    """
    current_features: list[Feature]
    precomputed_data_for_features: PopulationSampleWithFeaturesPrecomputedData
    complexity_array: np.ndarray

    expected_proportions: np.ndarray
    criteria_and_weights: list[(ScoringCriterion, float)]

    def __init__(self,
                 initial_features: list[Feature],
                 precomputed_sample_data,
                 complexity_function,
                 criteria_and_weights: list[(ScoringCriterion, float)],
                 expected_proportions=None):
        self.current_features = initial_features
        stripped_initial_features = [intermediate.to_legacy_feature() for intermediate in initial_features]
        self.precomputed_data_for_features = PopulationSampleWithFeaturesPrecomputedData(precomputed_sample_data,
                                                                                         stripped_initial_features)
        self.complexity_array = np.array([complexity_function(feature) for feature in stripped_initial_features])
        self.criteria_and_weights = criteria_and_weights
        self.expected_proportions = expected_proportions

    def get_complexity_array(self) -> np.ndarray:
        return utils.remap_array_in_zero_one(self.complexity_array)

    def get_explainability_array(self) -> np.ndarray:
        return 1.0 - self.get_complexity_array()

    def get_positive_fitness_correlation_array(self) -> np.ndarray:
        return utils.remap_array_in_zero_one(self.precomputed_data_for_features.get_average_fitness_vector())
        # return utils.remap_array_in_zero_one(self.precomputed_data_for_features.get_t_scores())

    def get_fitness_consistency_array(self) -> np.ndarray:
        t_scores = np.abs(self.precomputed_data_for_features.get_t_scores())
        return utils.remap_array_in_zero_one(t_scores)

    def get_negative_fitness_correlation_array(self) -> np.ndarray:
        return 1.0 - self.get_positive_fitness_correlation_array()

    def get_popularity_array(self):
        observed_proportions = self.precomputed_data_for_features.get_observed_proportions()
        chi_squareds = utils.chi_squared(observed_proportions, self.expected_proportions)
        which_are_good = np.sign(observed_proportions - self.expected_proportions)
        signed_chi_squareds = chi_squareds * which_are_good
        return utils.remap_array_in_zero_one(signed_chi_squareds)

    def get_phi_scores_for_each_feature(self) -> np.ndarray:
        count_for_each_value = (self.precomputed_data_for_features
                                .population_sample_precomputed
                                .count_for_each_var_val())

        observed_counts_for_each_feature = self.precomputed_data_for_features.count_for_each_feature

        amount_of_candidates = self.precomputed_data_for_features.sample_size

        def phi_score_for_feature(feature_index: int):
            used_vars = self.current_features[feature_index].get_used_variables()
            product_of_marginals = utils.product(count_for_each_value[var][val]
                                                 for var, val in used_vars)
            product_of_absences = utils.product((amount_of_candidates - count_for_each_value[var][val])
                                                for var, val in used_vars)

            n = amount_of_candidates
            n_all = observed_counts_for_each_feature[feature_index]

            if product_of_marginals * product_of_absences == 0:
                return 0.0
            else:
                return (n * n_all - product_of_marginals) / (np.sqrt(product_of_marginals * product_of_absences))

        return np.array([phi_score_for_feature(feature_index)
                         for feature_index in range(self.precomputed_data_for_features.amount_of_features)])

    def get_correlation_array(self) -> np.ndarray:
        return utils.remap_array_in_zero_one(self.get_phi_scores_for_each_feature())

    def get_anticorrelation_array(self) -> np.ndarray:
        return 1 - self.get_correlation_array()

    def get_instability_array(self):
        z_scores = self.precomputed_data_for_features.get_z_scores_compared_to_off_by_one()
        return utils.remap_array_in_zero_one(z_scores)

    def get_stability_array(self):
        return 1.0 - self.get_instability_array()

    def get_novelty_array(self):
        return 1.0 - self.get_popularity_array()

    def get_requested_atomic_score(self, criterion: ScoringCriterion) -> np.ndarray:
        if criterion == ScoringCriterion.EXPLAINABILITY:
            return self.get_explainability_array()
        elif criterion == ScoringCriterion.HIGH_FITNESS:
            return self.get_positive_fitness_correlation_array()
        elif criterion == ScoringCriterion.LOW_FITNESS:
            return self.get_negative_fitness_correlation_array()
        elif criterion == ScoringCriterion.FITNESS_CONSISTENCY:
            return self.get_fitness_consistency_array()
        elif criterion == ScoringCriterion.POPULARITY:
            return self.get_popularity_array()
        elif criterion == ScoringCriterion.NOVELTY:
            return self.get_novelty_array()
        elif criterion == ScoringCriterion.RESILIENCY:
            return self.get_stability_array()
        elif criterion == ScoringCriterion.CORRELATION:
            return self.get_correlation_array()
        elif criterion == ScoringCriterion.ANTICORRELATION:
            return self.get_anticorrelation_array()
        elif criterion == ScoringCriterion.INSTABILITY:
            return self.get_instability_array()
        else:
            raise Exception("The criterion for scoring was not specified")

    def invert_scores(self, scores: np.ndarray) -> np.ndarray:
        return 1.0 - scores

    def get_scores_of_features(self) -> np.ndarray:
        if len(self.criteria_and_weights) == 0:
            raise Exception("You requested a compound score composed of no criteria!")

        criteria, weights = utils.unzip(self.criteria_and_weights)
        individual_scores = [self.get_requested_atomic_score(criterion) for criterion in criteria]
        appropriately_inverted_scores = np.array([self.invert_scores(scores) if weight < 0 else scores
                                                  for weight, scores in zip(weights, individual_scores)])
        return utils.weighted_average_of_rows(appropriately_inverted_scores, np.abs(np.array(weights)))

    def get_the_best_features(self, how_many_to_keep: int) -> (list[Feature], np.ndarray):
        scores = self.get_scores_of_features()
        sorted_by_with_score = sorted(zip(self.current_features, scores), key=utils.second, reverse=True)
        features, scores_list = utils.unzip(sorted_by_with_score[:how_many_to_keep])
        return features, np.array(scores_list)


def criteria_and_weights_requires_expected_proportions(criteria_and_weights: list[(ScoringCriterion, float)]) -> bool:
    if len(criteria_and_weights) == 0:
        return False
    else:
        criteria, weights = utils.unzip(criteria_and_weights)
        special_criteria = [ScoringCriterion.NOVELTY, ScoringCriterion.POPULARITY]
        return any(special_criterion in criteria for special_criterion in special_criteria)


class FeatureDeveloper:
    """this class generates the useful, explainable features"""
    population_sample: PopulationSamplePrecomputedData
    previous_iterations: list[ParentPool]
    guaranteed_depth: int
    search_space: SearchSpace.SearchSpace
    complexity_function: Any  # SearchSpace.Feature -> float
    criteria_and_weights: [ScoringCriterion]
    amount_requested: int
    thoroughness: float

    def get_filter(self, intermediates: list[Feature], criteria_and_weights: list[(ScoringCriterion, float)]):
        expected_proportions = None
        if criteria_and_weights_requires_expected_proportions(criteria_and_weights):
            expected_proportions = np.array(
                [self.search_space.probability_of_feature_in_uniform(intermediate.to_legacy_feature())
                 for intermediate in intermediates])

        return FeatureFilter(intermediates,
                             self.population_sample,
                             self.complexity_function,
                             criteria_and_weights=criteria_and_weights,
                             expected_proportions=expected_proportions)

    def get_early_stages_criteria(self) -> list[(ScoringCriterion, float)]:
        return [(criterion, score) for criterion, score
                in self.criteria_and_weights if criterion == ScoringCriterion.EXPLAINABILITY]

    def get_just_explainability_filter(self, intermediates: list[Feature]):
        just_explainability = self.get_early_stages_criteria()
        # that is done so that the coefficient's sign is preserved
        return self.get_filter(intermediates, just_explainability)

    def get_trivial_parent_pool(self):
        trivial_features = Feature.get_all_trivial_features(self.search_space)
        scores = self.get_just_explainability_filter(trivial_features).get_scores_of_features()
        return ParentPool(trivial_features, scores)  # note how they don't get filtered!

    def __init__(self,
                 search_space,
                 population_sample,
                 guaranteed_depth,
                 extra_depth,
                 complexity_function,
                 criteria_and_weights: [ScoringCriterion],
                 amount_requested):
        self.search_space = search_space
        maximum_depth = self.search_space.dimensions

        self.population_sample = population_sample
        self.guaranteed_depth = min(guaranteed_depth, maximum_depth)
        self.extra_depth = min(extra_depth, maximum_depth)
        self.complexity_function = complexity_function
        self.criteria_and_weights = criteria_and_weights
        self.previous_iterations = [self.get_trivial_parent_pool()]
        self.amount_requested = amount_requested

    def get_parent_pool_of_weight(self, weight):
        return self.previous_iterations[weight - 1]

    @staticmethod
    def get_mixing_parent_weights(weight):
        parent_weight_1 = weight // 2
        parent_weight_2 = weight - parent_weight_1
        return parent_weight_1, parent_weight_2

    def get_amount_of_features_of_weight(self, weight):
        return len(self.get_parent_pool_of_weight(weight).features)

    def new_iteration(self,
                      amount_to_consider: int,
                      amount_to_return: int,
                      heuristic=False,
                      criteria_and_weights=None,
                      strict=True):
        new_weight = len(self.previous_iterations) + 1
        parent_weight_1, parent_weight_2 = FeatureDeveloper.get_mixing_parent_weights(new_weight)

        asexual_mixing = parent_weight_1 == parent_weight_2
        feature_mixer = FeatureMixer(self.get_parent_pool_of_weight(parent_weight_1),
                                     self.get_parent_pool_of_weight(parent_weight_2),
                                     asexual_mixing)
        if not strict:
            feature_mixer.disable_strict_mixing()
        if heuristic:
            considered_features = feature_mixer.get_heuristically_mixed_features(amount_to_consider)
        else:
            considered_features = feature_mixer.get_stochastically_mixed_features(amount_to_consider)

        if len(considered_features) == 0 and not strict:  # EMERGENCY! No good features could be generated
            # solution: allow to merge overlapping features
            print("Warning! The feature explorer ran out of merge-able features,"
                  " so strict merging was temporarily disabled")
            self.new_iteration(amount_to_consider, amount_to_return, heuristic, criteria_and_weights, strict=False)
            return
        feature_filter = self.get_filter(considered_features, criteria_and_weights)

        features, scores = feature_filter.get_the_best_features(amount_to_return)
        self.previous_iterations.append(ParentPool(features, scores))

    def get_schedule(self, strategy="heuristic where needed") -> list[(int,
                                                                       bool,
                                                                       list[(ScoringCriterion, float)],
                                                                       int,
                                                                       int)]:
        def get_total_amount_of_features(weight_category: int):
            """if all the variables had the same cardinality, this would be simpler"""
            return sum([utils.product(which_vars) for which_vars in
                        itertools.combinations(self.search_space.cardinalities, weight_category)])

        def get_likely_amount_of_important_features(weight_category: int):
            return self.amount_requested * self.search_space.total_cardinality

        def amount_to_keep_for_weight(weight_category: int):
            if weight_category <= self.guaranteed_depth:
                return get_total_amount_of_features(weight_category)
            else:
                return get_likely_amount_of_important_features(weight_category)

        def amount_to_consider_for_weight(weight_category: int, kept_in_category: int):
            if weight_category <= self.guaranteed_depth:
                return kept_in_category
            else:
                return kept_in_category * 2  # TODO think about this more

        def should_use_heuristic(weight_category: int) -> bool:
            if strategy == "heuristic where needed":
                return weight_category <= self.guaranteed_depth
            elif strategy == "always heuristic":
                return True
            elif strategy == "never heuristic":
                return False
            raise Exception(f"Requested an unknown strategy! ({strategy}")

        def which_criteria_to_use(weight_category: int):
            if weight_category <= self.guaranteed_depth - 1:
                return self.get_early_stages_criteria()
            else:
                return self.criteria_and_weights

        weights = list(range(2, self.extra_depth + 1))
        should_be_heuristic = [should_use_heuristic(weight) for weight in weights]
        which_criteria = [which_criteria_to_use(weight) for weight in weights]
        kepts = [amount_to_keep_for_weight(weight) for weight in weights]
        considereds = [amount_to_consider_for_weight(weight, kept) for weight, kept in zip(weights, kepts)]

        return zip(weights, should_be_heuristic, which_criteria, kepts, considereds)

    def develop_features(self, heuristic_strategy):
        settings_schedule = self.get_schedule(heuristic_strategy)

        for weight, use_heuristic, which_criteria, amount_to_keep, amount_to_consider in settings_schedule:
            print(
                f"In iteration where {weight = }: {use_heuristic = }, "
                f"{amount_to_keep = }, "
                f"{amount_to_consider = }, "
                f"{which_criteria = }")
            self.new_iteration(amount_to_consider,
                               amount_to_keep,
                               heuristic=use_heuristic,
                               criteria_and_weights=which_criteria)

    def get_developed_features(self) -> (list[SearchSpace.Feature], np.ndarray):
        """This is the function which returns the features you'll be using in the future!"""
        developed_features: list[Feature] = utils.concat_lists([parent_pool.features
                                                                for parent_pool in self.previous_iterations])
        feature_filterer = self.get_filter(developed_features, self.criteria_and_weights)

        return feature_filterer.get_the_best_features(self.amount_requested)


def find_features(problem: BenchmarkProblems.CombinatorialProblem.CombinatorialProblem,
                  sample_data: PopulationSamplePrecomputedData,
                  guaranteed_depth=2,
                  extra_depth=None,
                  criteria_and_weights=None,
                  amount_requested=12,
                  strategy="heuristic where needed") -> (list[SearchSpace.Feature], np.ndarray):
    if extra_depth is None:
        extra_depth = guaranteed_depth * 2

    feature_developer = FeatureDeveloper(search_space=problem.search_space,
                                         population_sample=sample_data,
                                         guaranteed_depth=guaranteed_depth,
                                         extra_depth=extra_depth,
                                         complexity_function=problem.get_complexity_of_feature,
                                         criteria_and_weights=criteria_and_weights,
                                         amount_requested=amount_requested)

    feature_developer.develop_features(heuristic_strategy=strategy)
    intermediate_features, scores = feature_developer.get_developed_features()
    raw_features = [intermediate.to_legacy_feature() for intermediate in intermediate_features]

    # give_explainability_and_average_fitnesses(raw_features, sample_data, problem, criteria)
    return raw_features, scores
