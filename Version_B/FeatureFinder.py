from typing import Any

import numpy as np
import random

import BenchmarkProblems.CombinatorialProblem
import SearchSpace
import HotEncoding
import utils
import Version_B.VariateModels
from Version_B.FeatureExplorer import IntermediateFeature, can_be_merged, merge_two_intermediate


class ParentPool:
    """this is a data structure to store a list of features and their scores"""
    """ The scores indicate how good a feature is, in terms of explainability and either fitness or novelty,
        These scores go from 0 to 1 and are also used to sample the features using weights.
        The main purpose of this class is to select a random feature and use it as a parent somewhere else
    """
    features: list[IntermediateFeature]
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

    def get_raw_features(self):
        return [intermediate.feature for intermediate in self.features]


class FeatureMixer:
    """ this class takes two sets of parents, and uses them to create new features"""
    """ it simply decides a parent from each set, and combines them"""
    """alternatively, you can use a greedy heuristic approach to get the best n"""
    parent_set_1: ParentPool
    parent_set_2: ParentPool
    """assumes parent_set_2 is either the same as parent_set_1 or bigger in len"""

    asexual: bool

    def __init__(self, parent_set_1: ParentPool, parent_set_2: ParentPool, asexual):
        if len(parent_set_1.features) < len(parent_set_2.features):
            self.parent_set_1 = parent_set_1
            self.parent_set_2 = parent_set_2
        else:
            self.parent_set_1 = parent_set_2
            self.parent_set_2 = parent_set_1

        self.asexual = asexual

    def select_parents(self) -> (IntermediateFeature, IntermediateFeature):
        return (self.parent_set_1.select_parent_randomly(), self.parent_set_2.select_parent_randomly())

    def create_random_feature(self) -> IntermediateFeature:
        while True:
            parent_1, parent_2 = self.select_parents()
            if can_be_merged(parent_1, parent_2):
                return merge_two_intermediate(parent_1, parent_2)

    def get_stochastically_mixed_features(self, amount: int) -> list[IntermediateFeature]:
        # TODO make this more efficient
        """this is the stochastic approach"""
        result = set()
        while len(result) < amount:
            result.add(self.create_random_feature())

        return list(result)

    def efficient_get_stochastically_mixed_features(self, amount: int) -> list[IntermediateFeature]:
        batch_size = amount
        result = set()

        def add_from_batch():
            batch_mothers = self.parent_set_1.select_n_parents_randomly(batch_size)
            batch_fathers = self.parent_set_2.select_n_parents_randomly(batch_size)
            offspring = [merge_two_intermediate(mother, father) for mother, father in zip(batch_mothers, batch_fathers)
                         if can_be_merged(mother, father)]
            result.update(offspring)

        while len(result) < amount:
            add_from_batch()

        return list(result)[:amount]

    @staticmethod
    def add_merged_if_mergeable(accumulator: set[IntermediateFeature],
                                mother: IntermediateFeature,
                                father: IntermediateFeature):
        if can_be_merged(mother, father):
            accumulator.add(merge_two_intermediate(mother, father))
            return True
        return False

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


class PopulationSamplePrecomputedData:
    search_space: SearchSpace.SearchSpace
    candidate_matrix: np.ndarray
    fitness_array: np.ndarray
    sample_size: int

    def __init__(self, search_space, population_sample, fitness_list):
        self.search_space = search_space
        self.candidate_matrix = HotEncoding.hot_encode_candidate_population(population_sample, self.search_space)
        self.fitness_array = np.array(fitness_list)
        self.sample_size = len(population_sample)

    @classmethod
    def from_problem(cls, problem: BenchmarkProblems.CombinatorialProblem.CombinatorialProblem,
                     amount_of_samples: int):
        search_space = problem.search_space
        population = [search_space.get_random_candidate() for _ in range(amount_of_samples)]
        fitnesses = [problem.score_of_candidate(candidate) for candidate in population]
        return cls(search_space, population, fitnesses)


class PopulationSampleWithFeaturesPrecomputedData:
    """this data structures stores matrices that are used around the other classes"""
    population_sample_precomputed: PopulationSamplePrecomputedData
    feature_presence_matrix: np.ndarray

    count_for_each_feature: np.ndarray
    complexity_array: np.ndarray

    def __init__(self, population_precomputed: PopulationSamplePrecomputedData,
                 raw_features: list[SearchSpace.Feature]):
        self.population_sample_precomputed = population_precomputed
        feature_matrix = HotEncoding.hot_encode_feature_list(raw_features,
                                                             self.population_sample_precomputed.search_space)

        self.feature_presence_matrix = Version_B.VariateModels.get_feature_presence_matrix_from_feature_matrix(
            self.population_sample_precomputed.candidate_matrix,
            feature_matrix)

        self.count_for_each_feature = np.sum(self.feature_presence_matrix, axis=0)

    def get_average_fitness_vector(self) -> np.ndarray:
        """returns the vector of average fitnesses for each feature"""
        sum_of_fitnesses = utils.weighted_sum_of_rows(self.feature_presence_matrix,
                                                      self.population_sample_precomputed.fitness_array)

        return np.where(self.count_for_each_feature == 0.0, 0.0, sum_of_fitnesses / self.count_for_each_feature)

    def get_overall_average_fitness(self):
        """returns the average fitness over the entire population"""
        return np.mean(self.population_sample_precomputed.fitness_array)

    def get_observed_proportions(self):
        """returns the observed proportion for every feature, from 0 to 1"""
        return self.count_for_each_feature / self.population_sample_precomputed.sample_size


class FeatureFilter:
    """this class accepts a list of features, and can
        * create scores based on their explainability + fitness / novelty
        * filter the given features based on those scores
    """
    current_features: list[IntermediateFeature]
    precomputed_data_for_features: PopulationSampleWithFeaturesPrecomputedData
    complexity_array: np.ndarray

    for_novelty: bool
    expected_proportions: np.ndarray

    def __init__(self,
                 initial_features,
                 precomputed_sample_data,
                 complexity_function,
                 importance_of_explainability,
                 expected_proportions=None):
        self.current_features = initial_features
        stripped_initial_features = [intermediate.feature for intermediate in initial_features]
        self.precomputed_data_for_features = PopulationSampleWithFeaturesPrecomputedData(precomputed_sample_data,
                                                                                         stripped_initial_features)
        self.complexity_array = np.array([complexity_function(feature) for feature in stripped_initial_features])
        self.importance_of_explainability = importance_of_explainability
        self.for_novelty = (expected_proportions is not None)
        self.expected_proportions = expected_proportions

    def get_explainability_array(self) -> np.ndarray:
        return (1.0 - utils.remap_array_in_zero_one(self.complexity_array)) ** 2

    def get_fitness_relevance_array(self) -> np.ndarray:
        average_fitnesses = self.precomputed_data_for_features.get_average_fitness_vector()
        average_overall_fitness = self.precomputed_data_for_features.get_overall_average_fitness()
        fitness_distance = np.abs(average_fitnesses - average_overall_fitness)  # TODO here use a t test instead
        return utils.remap_array_in_zero_one(fitness_distance) ** 2

    def get_novelty_array(self) -> np.ndarray:
        observed_proportions = self.precomputed_data_for_features.get_observed_proportions()
        distance_from_expected = np.abs(
            observed_proportions - self.expected_proportions)  # TODO here use a Chi squared metric instead
        return 1.0 - distance_from_expected

    def get_scores_of_features(self, with_criteria = True) -> np.ndarray:
        explainabilities = self.get_explainability_array()

        if not with_criteria:
            return explainabilities

        criteria_scores = self.get_novelty_array() if self.for_novelty else self.get_fitness_relevance_array()
        return utils.geometric_weighted_average(explainabilities, self.importance_of_explainability,
                                  criteria_scores, 1.0 - self.importance_of_explainability)

    def get_the_best_features(self, how_many_to_keep: int, with_criteria = True) -> (list[IntermediateFeature], np.ndarray):
        scores = self.get_scores_of_features(with_criteria)

        sorted_by_with_score = sorted(zip(self.current_features, scores), key=utils.second, reverse=True)
        features, scores_list = utils.unzip(sorted_by_with_score[:how_many_to_keep])
        return features, np.array(scores_list)


class FeatureDeveloper:
    """this class generates the useful, explainable features"""
    population_sample: PopulationSamplePrecomputedData
    previous_iterations: list[ParentPool]
    depth: int
    search_space: SearchSpace.SearchSpace
    complexity_function: Any  # SearchSpace.Feature -> float
    importance_of_explainability: float
    for_novelty: bool

    def get_filter(self, intermediates: list[IntermediateFeature]):
        # TODO: in the future you might want the expected proportions to be obtained from previous iterations of the GA!
        # For now, they are obtained as if it always was the first generation (and the one before was uniformly random)
        if self.for_novelty:
            expected_proportions = np.ndarray([self.search_space.probability_of_feature_in_uniform(intermediate.feature)
                                               for intermediate in intermediates])
            return FeatureFilter(intermediates,
                                 self.population_sample,
                                 self.complexity_function,
                                 self.importance_of_explainability,
                                 expected_proportions)
        else:
            return FeatureFilter(intermediates,
                                 self.population_sample,
                                 self.complexity_function,
                                 self.importance_of_explainability,
                                 expected_proportions=None)

    def get_trivial_parent_pool(self):
        trivial_features = [IntermediateFeature.get_trivial_feature(var, val)
                            for var, val in self.search_space.get_all_var_val_pairs()]

        feature_filter = self.get_filter(trivial_features)
        scores = feature_filter.get_scores_of_features(with_criteria=False)

        return ParentPool(trivial_features, scores)  # note how they don't get filtered!

    def __init__(self,
                 search_space,
                 population_sample,
                 depth,
                 complexity_function,
                 importance_of_explainability,
                 for_novelty=False):
        self.search_space = search_space
        self.population_sample = population_sample
        self.depth = depth
        self.complexity_function = complexity_function
        self.importance_of_explainability = importance_of_explainability
        self.for_novelty = for_novelty

        self.previous_iterations = [self.get_trivial_parent_pool()]

    def get_parent_pool_of_weight(self, weight):
        return self.previous_iterations[weight - 1]

    def new_iteration(self, amount_to_consider: int, amount_to_return: int, heuristic=False, with_criteria=True):
        new_weight = len(self.previous_iterations) + 1
        parent_weight_1 = new_weight // 2
        parent_weight_2 = new_weight - parent_weight_1

        asexual_mixing = parent_weight_1 == parent_weight_2
        feature_mixer = FeatureMixer(self.get_parent_pool_of_weight(parent_weight_1),
                                     self.get_parent_pool_of_weight(parent_weight_2),
                                     asexual_mixing)

        if heuristic:
            considered_features = feature_mixer.get_heuristically_mixed_features(amount_to_consider)
        else:
            considered_features = feature_mixer.efficient_get_stochastically_mixed_features(amount_to_consider)

        feature_filter = self.get_filter(considered_features)

        features, scores = feature_filter.get_the_best_features(amount_to_return, with_criteria=with_criteria)
        self.previous_iterations.append(ParentPool(features, scores))



    def how_many_features_to_consider_in_weight_category(self, weight):
        average_cardinality = self.search_space.total_cardinality / self.search_space.dimensions
        total_amount_in_weight_category = utils.binomial_coeff(self.search_space.dimensions, weight) * (
            average_cardinality ** weight)
        return utils.binomial_coeff(self.search_space.dimensions, weight) * average_cardinality * weight


    def develop_features(self, heuristic=False):

        def should_use_heuristic(iteration):
            heuristic_threshold = self.depth * 0.8
            if heuristic == "initially":
                return iteration < heuristic_threshold
            else:
                return heuristic

        def should_use_criteria(iteration):
            criteria_threshold = self.depth // 2
            return iteration >= criteria_threshold

        # TODO choose good numbers
        for i in range(self.depth-1):
            use_heuristic = should_use_heuristic(i)
            use_criteria = should_use_criteria(i)

            weight = i+2
            total_amount_in_weight_category = self.how_many_features_to_consider_in_weight_category(i+2)
            amount_to_keep_per_category = int(self.search_space.total_cardinality ** ((i+2) / 2))

            if use_heuristic:
                amount_to_consider = amount_to_keep_per_category ** 2
            else:
                amount_to_consider = amount_to_keep_per_category


            print(f"On the {i}th loop of develop_features, {use_heuristic = },"
                  f" {use_criteria = }, "
                  f"{amount_to_keep_per_category = }, "
                  f"{amount_to_consider = }")
            self.new_iteration(amount_to_consider,
                               amount_to_keep_per_category,
                               heuristic=use_heuristic,
                               with_criteria=use_criteria)

    def get_developed_features(self) -> (list[SearchSpace.Feature], np.ndarray):
        """This is the function which returns the features you'll be using in the future!"""
        developed_features: list[IntermediateFeature] = utils.concat([parent_pool.features
                                                                      for parent_pool in self.previous_iterations])
        feature_filterer = self.get_filter(developed_features)

        return feature_filterer.get_the_best_features(self.search_space.total_cardinality)


def find_features(problem: BenchmarkProblems.CombinatorialProblem.CombinatorialProblem,
                  depth: int,
                  importance_of_explainability: float,
                  sample_size: int,
                  for_novelty: bool) -> (list[SearchSpace.Feature], np.ndarray):
    sample_data = PopulationSamplePrecomputedData.from_problem(problem, sample_size)
    feature_developer = FeatureDeveloper(search_space=problem.search_space,
                                         population_sample=sample_data,
                                         depth=depth,
                                         complexity_function=problem.get_complexity_of_feature,
                                         importance_of_explainability=importance_of_explainability,
                                         for_novelty=for_novelty)

    feature_developer.develop_features(heuristic=True)

    intermediate_features, scores = feature_developer.get_developed_features()
    raw_features = [intermediate.feature for intermediate in intermediate_features]

    give_explainability_and_average_fitnesses(raw_features, sample_data, problem)
    return raw_features, scores



def give_explainability_and_average_fitnesses(features: list[SearchSpace.Feature],
                                              population_sample: PopulationSamplePrecomputedData,
                                              problem: BenchmarkProblems.CombinatorialProblem.CombinatorialProblem):
    sample_with_features = PopulationSampleWithFeaturesPrecomputedData(population_sample, features)
    fitnesses = sample_with_features.get_average_fitness_vector()
    complexities = np.array([problem.get_complexity_of_feature(feature) for feature in features])

    for feature, fitness, complexity in zip(features, fitnesses, complexities):
        problem.pretty_print_feature(feature)
        print(f"Has {fitness = :.2f}, {complexity = :.2f}")


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
