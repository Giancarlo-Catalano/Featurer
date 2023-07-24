import math

import numpy as np

import SearchSpace
import HotEncoding
import utils

from Version_B import VariateModels


class FeatureDiscoverer:
    search_space: SearchSpace.SearchSpace
    hot_encoder: HotEncoding.HotEncoder

    # candidate_matrix: matrix where each row is a candidateH
    # fitness_scores: np.array of scalar fitnesses for each candidate, in the same order
    # merging_power: int, >=2
    # complexity_function: a function candidateC->complexity_score, where complexity_score in [0, 1]

    # features: set of featureH

    def __init__(self, search_space, candidateC_population, fitness_scores, merging_power, complexity_function,
                 complexity_damping=2,
                 importance_of_explainability=0.5):
        self.expected_frequency_of_features = None
        self.explainability_of_features = None
        self.explainable_features = None
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.amount_of_candidates = len(candidateC_population)
        self.candidate_matrix = self.hot_encoder.to_hot_encoded_matrix(candidateC_population)
        self.fitness_of_candidates = np.array(fitness_scores)
        self.merging_power = merging_power
        self.complexity_function = complexity_function
        self.importance_of_explainability = importance_of_explainability
        self.complexity_damping = complexity_damping

        self.trivial_featuresH = self.hot_encoder.get_hot_encoded_trivial_features()
        self.flat_clash_matrix = HotEncoding.get_search_space_flat_clash_matrix(self.search_space)

        self.variate_model_builder = VariateModels.VariateModels(self.search_space)

    @property
    def importance_of_fitness(self):
        return 1.0 - self.importance_of_explainability

    def select_valid_features(self, input_features):
        """removes the features which are invalid because of toe-stepping"""
        clash_results = HotEncoding.fast_features_are_invalid(np.array(input_features), self.flat_clash_matrix)

        return [feature for (feature, clashes)
                in zip(input_features, clash_results)
                if not clashes]

    def get_explainability_threshold_to_filter_features(self, min_complexity, max_complexity, current_size, ideal_size):
        """We intend to filter the features to keep a certain amount of them,
        This returns a threshold on the explainability that can be used to achieve this effect, approximately."""

        # assumes that current size is greater than ideal size
        harshness = (ideal_size / current_size) ** (self.merging_power * 2)
        threshold = utils.weighted_sum(min_complexity, 1 - harshness,
                                       max_complexity, harshness)
        return threshold

    def cull_organised_by_weight(self, organised_by_weight):
        """ in explore_features, we need to cull the features to keep the most explainable ones"""
        current_size = np.sum([len(weight_category) for weight_category in organised_by_weight])
        ideal_size = self.search_space.total_cardinality ** (math.log2(self.merging_power) + 1)
        # TODO find a good default value, and set this from the outside 
        #  (perhaps use a "urgency" parameter, I've got notes back home!)

        # if it's already small enough, you can use the original
        if current_size <= ideal_size:
            return organised_by_weight

        def complexity_of_featureH(featureH):
            return self.get_complexity_of_feature(self.hot_encoder.feature_from_hot_encoding(featureH))

        # calculate the complexities once, and use them to 1) calculate the range, 2) to filter
        complexities_organised_by_weight = [np.array([complexity_of_featureH(fH) for fH in weight_category])
                                            for weight_category in organised_by_weight]

        # calculating the range and then the threshold
        (min_complexity, max_complexity) = utils.get_min_max_of_arrays(complexities_organised_by_weight)

        threshold = self.get_explainability_threshold_to_filter_features(min_complexity, max_complexity,
                                                                         current_size, ideal_size)

        def cull_weight_category(original_items, complexities_in_category):
            return [from_original for (from_original, complexity) in zip(original_items, complexities_in_category)
                    if complexity <= threshold]

        # cull each category
        return [cull_weight_category(original, complexities) for (original, complexities)
                in zip(organised_by_weight, complexities_organised_by_weight)]

    def explore_features(self, at_most):
        """returns *all* the *valid* merges obtained by choosing at_most features from feature_pool"""
        organised_by_weight = list(list() for _ in range(at_most + 1))
        organised_by_weight[0].append(self.hot_encoder.empty_feature)

        def consider_feature(trivial_feature):
            for weight_category in reversed(range(at_most)):  # NOTE: the lack of +1
                if len(organised_by_weight[weight_category]) == 0:
                    continue  # grrr I hate this keyword... why not call it skip?

                new_features = np.array([HotEncoding.merge_features(old_feature, trivial_feature)
                                         for old_feature in organised_by_weight[weight_category]])

                new_features = self.select_valid_features(new_features)
                # new_features = self.select_simple_features(new_features)

                organised_by_weight[weight_category + 1].extend(new_features)

        for feature in self.trivial_featuresH:
            consider_feature(feature)
            organised_by_weight = self.cull_organised_by_weight(organised_by_weight)

        return utils.concat(
            organised_by_weight[1:])

    def get_complexity_of_feature(self, featureC):
        raw_complexity = self.complexity_function(featureC)
        dampened_complexity = (1.0 / self.complexity_damping) * (raw_complexity - 0.5) + 0.5
        return dampened_complexity

    def get_explainability_of_feature(self, featureC):
        """ returns a score in [0,1] describing how explainable the feature is,
                based on the given complexity function"""
        return 1.0 - self.get_complexity_of_feature(featureC)

    def get_expected_frequency_of_feature(self, featureC):  # this might be replaced from the outside in the future
        return self.search_space.probability_of_feature_in_uniform(featureC)

    def generate_explainable_features(self):
        self.explainable_features = self.explore_features(at_most=self.merging_power)
        explainable_featuresC = [self.hot_encoder.feature_from_hot_encoding(fH) for fH in self.explainable_features]
        self.explainability_of_features = np.array(
            [self.get_explainability_of_feature(fC) for fC in explainable_featuresC])
        self.expected_frequency_of_features = np.array(
            [self.get_expected_frequency_of_feature(fC) * self.amount_of_candidates for fC in explainable_featuresC])

    def get_weighed_sum_with_explainability_scores(self, good_and_bad_scores):
        (goodness_scores, badness_scores) = good_and_bad_scores
        goodness_weighted_sum = self.explainability_of_features * self.importance_of_explainability \
                                + goodness_scores * self.importance_of_fitness

        badness_weighted_sum = self.explainability_of_features * self.importance_of_explainability \
                               + badness_scores * self.importance_of_fitness
        return (goodness_weighted_sum, badness_weighted_sum)

    def get_explainable_features(self, criteria='fitness'):
        amount_to_keep = self.search_space.total_cardinality

        def get_best_using_scores(criteria_scores):
            with_scores = list(zip(self.explainable_features, criteria_scores))
            with_scores.sort(key=utils.second, reverse=True)
            return with_scores[:amount_to_keep]

        if criteria == 'all':
            return self.explainable_features

        scores = None
        feature_presence_matrix = self.variate_model_builder \
            .get_feature_presence_matrix(self.candidate_matrix, self.explainable_features)
        if criteria == 'fitness':
            scores = self.variate_model_builder \
                .get_fitness_unfitness_scores(feature_presence_matrix,
                                              self.fitness_of_candidates)
        elif criteria == 'popularity':
            scores = self.variate_model_builder \
                .get_popularity_unpopularity_scores(feature_presence_matrix,
                                                    self.expected_frequency_of_features)

        (goodness, badness) = self.get_weighed_sum_with_explainability_scores(scores)

        return (get_best_using_scores(goodness), get_best_using_scores(badness))
