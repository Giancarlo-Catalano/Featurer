import numpy as np

import SearchSpace
import HotEncoding
import Version_B.VariateModels
import utils


class IntermediateFeature:
    """A feature, with added information on leftmost and rightmost set parameter"""
    start: int
    end: int
    feature: SearchSpace.Feature

    def __init__(self, start, end, feature):
        self.start = start
        self.end = end
        self.feature = feature

    def __repr__(self):
        return f"IntermediateFeature(start = {self.start}, end = {self.end}, feature = {self.feature})"

    @classmethod
    def get_trivial_feature(cls, var, val):
        return cls(var, var, SearchSpace.Feature.trivial_feature(var, val))


    def __hash__(self):
        return self.feature.__hash__()

    def __eq__(self, other):
        return self.feature.__eq__(other.feature)


def merge_two_intermediate(left: IntermediateFeature, right: IntermediateFeature):
    """creates the union of two intermediate features"""
    # NOTE: the order of the arguments matters!
    new_start = min(left.start, right.start)
    new_end = max(left.end, right.end)
    new_feature = SearchSpace.merge_two_features(left.feature, right.feature)

    return IntermediateFeature(new_start, new_end, new_feature)


def can_be_merged(a: IntermediateFeature, b: IntermediateFeature):
    """Returns true if the 2 features can be merged without overlaps"""
    return (a.end < b.start) or (b.end < a.start)


class IntermediateFeatureGroup:
    """Data structure to manage a group of features of the same weight"""
    intermediate_features: set  # of intermediate features
    weight: int  # all the intermediate features in the group have the same weight.

    def __init__(self, intermediate_features, weight):
        self.intermediate_features = intermediate_features
        self.weight = weight

    def __repr__(self):
        result = f"IntermediateFeatureGroup(weight = {self.weight}):"
        for intermediate_feature in self.intermediate_features:
            result += f"\n\t{intermediate_feature}"

        return result

    @classmethod
    def get_0_weight_group(cls):
        return cls(set(), 0)

    @classmethod
    def get_trivial_weight_group(cls, search_space: SearchSpace.SearchSpace):
        weight = 1
        var_vals = search_space.get_all_var_val_pairs()
        trivial_features = set(IntermediateFeature.get_trivial_feature(var, val)
                               for (var, val) in var_vals)
        return cls(trivial_features, weight)

def cull_by_complexity(intermediate_features, complexity_function, wanted_size) -> set:
    if wanted_size > len(intermediate_features):
        return intermediate_features
    with_complexity = [(intermediate, complexity_function(intermediate.feature))
                       for intermediate in intermediate_features]

    with_complexity.sort(key=utils.second)
    return set(intermediate for (intermediate, score) in with_complexity[:wanted_size])


def mix_intermediate_feature_groups(first_group: IntermediateFeatureGroup, second_group: IntermediateFeatureGroup):
    new_weight = first_group.weight + second_group.weight
    new_elements = []

    for from_first in first_group.intermediate_features:
        for from_second in second_group.intermediate_features:
            if can_be_merged(from_first, from_second):
                new_elements.append(merge_two_intermediate(from_first, from_second))

    return IntermediateFeatureGroup(new_elements, new_weight)


class GroupManager:
    groups_by_weight: list  # groups_by_weight[n] is the group of weight n
    ideal_size_of_group: int

    def __init__(self, search_space: SearchSpace.SearchSpace):
        self.groups_by_weight = [IntermediateFeatureGroup.get_0_weight_group(),
                                 IntermediateFeatureGroup.get_trivial_weight_group(search_space)]
        self.ideal_size_of_group = max(search_space.total_cardinality*search_space.dimensions, 100)
        # this is because very small problems will attempt culling and erase a significant chunk of the entire space

    def get_group(self, weight) -> IntermediateFeatureGroup:
        return self.groups_by_weight[weight]

    def get_necessary_weights_to_obtain(self, target_weight):
        half_total = target_weight // 2
        return (half_total, half_total + (target_weight % 2))

    def make_incremented_weight_category(self, feature_complexity_function):
        new_weight = len(self.groups_by_weight)
        left_weight, right_weight = self.get_necessary_weights_to_obtain(new_weight)

        new_group = mix_intermediate_feature_groups(
            self.get_group(left_weight), self.get_group(right_weight))

        new_group.intermediate_features = cull_by_complexity(new_group.intermediate_features, feature_complexity_function, self.ideal_size_of_group)
        self.groups_by_weight.append(new_group)


def develop_groups_to_weight(search_space: SearchSpace.SearchSpace, max_weight: int, feature_complexity_function) -> GroupManager:
    current_group_manager = GroupManager(search_space)
    for _ in range(2, max_weight+1):
        current_group_manager.make_incremented_weight_category(feature_complexity_function)

    return current_group_manager


def retrieve_explainable_features(search_space: SearchSpace.SearchSpace, max_weight: int, feature_complexity_function):
    groups: GroupManager = develop_groups_to_weight(search_space, max_weight, feature_complexity_function)

    result = set()
    for weight_category in groups.groups_by_weight[1:]:
        result.update(weight_category.intermediate_features)

    # result = cull_by_complexity(result, feature_complexity_function, len(result))
    return [intermediate.feature for intermediate in result]

class FeatureExplorer:
    search_space: SearchSpace.SearchSpace
    hot_encoder: HotEncoding.HotEncoder
    merging_power: int
    variate_model_generator: Version_B.VariateModels.VariateModels

    def __init__(self, search_space, merging_power, complexity_function,
                 importance_of_explainability=0.5):

        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.merging_power = merging_power
        self.importance_of_explainability = importance_of_explainability
        self.complexity_function = complexity_function

        self.variate_model_generator = Version_B.VariateModels.VariateModels(self.search_space)

        self.explanainable_features = retrieve_explainable_features(self.search_space,
                                                                    self.merging_power,
                                                                    self.complexity_function)

    def get_complexity_of_featureC(self, featureC):
        return self.complexity_function(featureC)


    def get_average_fitnesses_and_frequencies(self, candidateC_population, fitness_list, features):
        candidate_matrix = self.hot_encoder.to_hot_encoded_matrix(candidateC_population)
        featuresH = [self.hot_encoder.feature_to_hot_encoding(featureC) for featureC in features]
        feature_presence_matrix = self.variate_model_generator.get_feature_presence_matrix(candidate_matrix, featuresH)
        fitness_array = np.array(fitness_list)

        average_fitnesses = self.variate_model_generator.\
            get_average_fitness_of_features_from_matrix(feature_presence_matrix, fitness_array)
        frequencies = self.variate_model_generator.get_observed_frequency_of_features(feature_presence_matrix)

        return average_fitnesses, frequencies

    def classify_features_by_prodigy(self, candidateC_population, fitness_list):
        average_fitnesses, frequencies = self.get_average_fitnesses_and_frequencies(candidateC_population,
                                                                                    fitness_list,
                                                                                    self.explanainable_features)
        average_overall_fitness = np.mean(fitness_list)
        expected_frequencies = np.array([self.search_space.probability_of_feature_in_uniform(featureC)
                                for featureC in self.explanainable_features])*len(fitness_list)

        fit_features, unfit_features, popular_features, unpopular_features = [], [], [], []
        fit_scores, unfit_scores, popular_scores, unpopular_scores = [], [], [], []

        for featureC, average_fitness, observed_frequency, expected_frequency in zip(self.explanainable_features, average_fitnesses, frequencies, expected_frequencies):
            is_fit = average_fitness > average_overall_fitness
            is_popular = observed_frequency > expected_frequency
            fitness_significance = abs(average_fitness-average_overall_fitness)
            frequency_significance = utils.chi_squared(observed_frequency, expected_frequency)

            if is_fit:
                fit_features.append(featureC)
                fit_scores.append(fitness_significance)
            else:
                unfit_features.append(featureC)
                unfit_scores.append(fitness_significance)

            if is_popular:
                popular_features.append(featureC)
                popular_scores.append(frequency_significance)
            else:
                unpopular_features.append(featureC)
                unpopular_scores.append(frequency_significance)

        return ((fit_features, fit_scores),
                (unfit_features, unfit_scores),
                (popular_features, popular_scores),
                (unpopular_features, unpopular_scores))

    def combine_prodigy_score_with_explainabilities(self, features_and_prodigy_scores):
        """each feature has a criteria score (fitness, unfitness, pop, unpop), and an explainability"""
        """this function will combine them using a simple interpolation average"""
        """this is where 'importance of explainability' gets used!'"""
        (features, scores) = features_and_prodigy_scores
        explainabilities = 1.0-utils.remap_array_in_zero_one([self.get_complexity_of_featureC(feature) for feature in features])
        score_array = utils.remap_array_in_zero_one(np.array(scores))


        # debug
        # print("In the given list of features, the values are as follows:")
        # for feature, explainability, criteria_score in zip(features, explainabilities, score_array):
        #     print(f"For the feature {feature}, expl = {explainability:.2f}, score = {criteria_score:.2f}")
        # end of debug

        return zip(features, utils.weighted_sum(explainabilities, self.importance_of_explainability,
                                  score_array, 1.0-self.importance_of_explainability))

    def get_important_explainable_features(self, candidateC_population, fitness_list):
        fits, unfits, populars, unpopulars = self.classify_features_by_prodigy(candidateC_population, fitness_list)

        fit_prodigies = self.combine_prodigy_score_with_explainabilities(fits)
        unfit_prodigies = self.combine_prodigy_score_with_explainabilities(unfits)
        popular_prodigies = self.combine_prodigy_score_with_explainabilities(populars)
        unpopular_prodigies = self.combine_prodigy_score_with_explainabilities(unpopulars)

        def sort_and_filter_by_criteria(zipped_prodigies_with_scores):
            how_many_to_keep = self.search_space.total_cardinality
            sorted_by_criteria = sorted(zipped_prodigies_with_scores, key=utils.second, reverse=True)
            above_ten_percent = [(feature, score) for feature, score in sorted_by_criteria if score > 0.1]
            return above_ten_percent[:how_many_to_keep]

        fit_prodigies = sort_and_filter_by_criteria(fit_prodigies)
        unfit_prodigies = sort_and_filter_by_criteria(unfit_prodigies)
        popular_prodigies = sort_and_filter_by_criteria(popular_prodigies)
        unpopular_prodigies = sort_and_filter_by_criteria(unpopular_prodigies)

        return fit_prodigies, unfit_prodigies, popular_prodigies, unpopular_prodigies






