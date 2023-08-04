import math

import numpy as np

import SearchSpace
import HotEncoding
import Version_B.VariateModels
import utils

from Version_B import VariateModels


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
    def get_trivial_feature(cls, var, val, search_space: SearchSpace.SearchSpace):
        return cls(var, var, search_space.get_single_value_feature(var, val))


def merge_two_intermediate(left: IntermediateFeature, right: IntermediateFeature):
    """creates the union of two intermediate features"""
    # NOTE: the order of the arguments matters!
    new_start = left.start
    new_end = right.end
    new_feature = SearchSpace.merge_two_features(left.feature, right.feature)

    return IntermediateFeature(new_start, new_end, new_feature)


def can_be_merged(left: IntermediateFeature, right: IntermediateFeature):
    """Returns true if the 2 features can be merged without overlaps"""
    """Additionally, it only allows two features to be merged in one way (ie AB is allowed, BA is not)"""
    # NOTE: the order of the arguments matters!
    return left.end < right.start


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
        trivial_features = set(IntermediateFeature.get_trivial_feature(var, val, search_space)
                               for (var, val) in var_vals)
        return cls(trivial_features, weight)


    def cull_by_complexity(self, complexity_function, wanted_size):
        if wanted_size > len(self.intermediate_features):
            return
        with_complexity = [(intermediate, complexity_function(intermediate.feature))
                           for intermediate in self.intermediate_features]

        with_complexity.sort(key=utils.second)
        self.intermediate_features = set(intermediate for (intermediate, score) in with_complexity[:wanted_size])


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
        self.ideal_size_of_group = search_space.total_cardinality*search_space.dimensions

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

        new_group.cull_by_complexity(feature_complexity_function, self.ideal_size_of_group)
        self.groups_by_weight.append(new_group)


def develop_groups_to_weight(search_space: SearchSpace.SearchSpace, max_weight: int, feature_complexity_function) -> GroupManager:
    current_group_manager = GroupManager(search_space)
    for _ in range(2, max_weight+1):
        current_group_manager.make_incremented_weight_category(feature_complexity_function)

    return current_group_manager


def get_all_features_of_weight_at_most(search_space: SearchSpace.SearchSpace, max_weight: int, feature_complexity_function):
    groups: GroupManager = develop_groups_to_weight(search_space, max_weight, feature_complexity_function)

    result = []
    for weight_category in groups.groups_by_weight[1:]:
        result.extend(intermediate.feature for intermediate in weight_category.intermediate_features)

    return result

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
        self.importance_of_explainability = 0.5

        self.variate_model_generator = Version_B.VariateModels.VariateModels(self.search_space)

    @property
    def importance_of_fitness(self):
        return 1.0 - self.importance_of_explainability

    def get_complexity_of_featureC(self, featureC):
        return self.complexity_function(featureC)

    def get_explainability_of_feature(self, featureC):
        """ returns a score in [0,1] describing how explainable the feature is,
                based on the given complexity function"""
        return 1.0 / self.get_complexity_of_featureC(featureC)


    def get_explainabilities(self, features):
        return np.array([self.get_explainability_of_feature(feature) for feature in features])



    # TODO

    # "Train the model" to recognise popular, unpopular, fit and unfit features
    # first, we pass a candidate population, with fitnesses
    # from that we obtain the feature presence matrix, stored in self
    # using the feature presence matrix and the fitnesses we can
    #   calculate the average fitnesses, check which they are greater than expected, then force [0, 1]
    #   calculate the frequency, check when they are greater than expected, then force [0, 1]
    # using these scores, determine which features are fit (fitness average is high) etc,
    # separate the fit and unfit features (they should be disjoint), and the pop and unpop
    # calculate their weighted averages with the explainabilities.

