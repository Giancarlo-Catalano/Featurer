import numpy as np

import SearchSpace
import HotEncoding
import utils


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
        self.explainability_of_features = None
        self.explainable_features = None
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.candidate_matrix = self.hot_encoder.to_hot_encoded_matrix(candidateC_population)
        self.fitness_scores = np.array(fitness_scores)
        self.merging_power = merging_power
        self.complexity_function = complexity_function
        self.importance_of_explainability = importance_of_explainability
        self.complexity_damping = complexity_damping

        self.trivial_featuresH = self.hot_encoder.get_hot_encoded_trivial_features()
        self.flat_clash_matrix = HotEncoding.get_search_space_flat_clash_matrix(self.search_space)

    @property
    def importance_of_fitness(self):
        return 1.0 - self.importance_of_explainability

    def select_valid_features(self, input_features):
        """removes the features which are invalid because of toestepping"""
        clash_results = HotEncoding.fast_features_are_invalid(np.array(input_features), self.flat_clash_matrix)

        return [feature for (feature, clashes)
                in zip(input_features, clash_results)
                if not clashes]

    def select_simple_features(self, input_features):
        """returns the simpler of the input features"""
        ideal_size = self.search_space.total_cardinality * self.merging_power
        current_size = len(input_features)

        if current_size <= ideal_size:
            # print(f"A list of {len(featureH_list)} was spared")
            return input_features

        complexity_scores = np.array([self.complexity_function(self.hot_encoder.feature_from_hot_encoding(featureH))
                                      for featureH in input_features])

        # we then select the top importance_of_explainability of the population, scored by explainability
        harshness = (ideal_size / current_size) ** (self.merging_power * 2)
        threshold = utils.weighted_sum(np.min(complexity_scores), 1 - harshness,
                                       np.max(complexity_scores), harshness)

        return [featureH for (featureH, complexity) in zip(input_features, complexity_scores)
                if complexity <= threshold]

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
                organised_by_weight[weight_category + 1] = \
                    self.select_simple_features(organised_by_weight[weight_category + 1])

        for feature in self.trivial_featuresH:
            consider_feature(feature)

        return utils.concat(
            organised_by_weight[1:])

    def get_explainability_of_feature(self, featureC):
            """ returns a score in [0,1] describing how explainable the feature is,
                based on the given complexity function"""
            raw_complexity = self.complexity_function(featureC)
            dampened_complexity = (1.0 / self.complexity_damping) * (raw_complexity - 0.5) + 0.5
            return 1.0 - dampened_complexity

    def get_expected_frequency_of_feature(self, featureC):  # this might be replaced from the outside in the future
        return self.search_space.probability_of_feature_in_uniform(featureC)

    def generate_explainable_features(self):
        self.explainable_features = self.explore_features(at_most=self.merging_power)
        explainable_featuresC = [self.hot_encoder.feature_from_hot_encoding(fH) for fH in self.explainable_features]
        self.explainability_of_features = np.array([self.get_explainability_of_feature(fC) for fC in explainable_featuresC])
        self.expected_frequency_of_features = np.array([self.get_expected_frequency_of_feature(fC) for fC in explainable_featuresC])

    @staticmethod
    def boost_range(x):
        """the input array is in [0, 1], and the result will have values lower than 0.5 lowered, greater than 0.5 increased"""
        return 3 * x ** 2 - 2 * x ** 3  # this is the solution to integral([x*(1-x)]^k) for k=1

    def which_candidates_contain_which_features(self, featureH_pool):
        """returns a matrix in which element i, j is 1 if candidate i contains feature j"""
        feature_matrix = np.transpose(np.array(featureH_pool))
        positive_when_absent = (1 - self.candidate_matrix) @ feature_matrix
        return 1 - np.minimum(positive_when_absent, 1)

    def get_average_fitnesses_for_features(self, featureH_pool):
        """returns an array with the scores of the given features, adjusted by complexity and statistical significance"""

        feature_presence_matrix = self.which_candidates_contain_which_features(featureH_pool)
        count_for_each_feature = np.sum(feature_presence_matrix, axis=0)
        sum_of_fitness_for_each_feature = np.sum(feature_presence_matrix * utils.to_column_vector(self.fitness_scores),
                                                 axis=0)

        average_fitnesses = np.array([total / float(count) if count > 0 else 0.0 for total, count in
                                      zip(sum_of_fitness_for_each_feature, count_for_each_feature)])

        # then we remap the fitnesses to be between 0 and 1
        unboosted = utils.remap_array_in_zero_one(average_fitnesses)  # forces them to be between 0 and 1
        return self.boost_range(unboosted)

    def get_fitness_unfitness_scores(self):
        """returns a pair (score_goodness, score_badness), where the goodness is fitness if on_commonality = False"""
        """if on_commonality is True, goodness is the popularity in the candidate population"""
        fitness_percentiles = self.get_average_fitnesses_for_features(self.explainable_features)
        return (fitness_percentiles, 1 - fitness_percentiles)

    def get_frequency_of_each_feature(self, featureH_pool):
        """returns an array indicating how common each feature is, as a percentage in [0, 1]"""

        feature_presence_matrix = self.which_candidates_contain_which_features(featureH_pool)
        count_for_each_feature = np.sum(feature_presence_matrix, axis=0)
        return utils.remap_array_in_zero_one(count_for_each_feature)

    def get_popularity_unpopularity_scores(self):
        """returns a pair (list_of_good, list_of_bad), where the lists contain features of size at most self.merging_power"""
        observed_frequencies = self.get_frequency_of_each_feature(self.explainable_features)
        expected_frequencies = self.expected_frequency_of_features

        goodness_badness_pairs = [(x_2, 0.0) if is_popular else (0.0, x_2)
                                  for (x_2, is_popular)
                                  in zip(utils.chi_squared(observed_frequencies, expected_frequencies),
                                         observed_frequencies > expected_frequencies)]

        (goodness_scores, badness_scores) = utils.unzip(goodness_badness_pairs)
        return (goodness_scores, badness_scores)

    def get_weighed_sum_with_explainability_scores(self, good_and_bad_scores):
        (goodness_scores, badness_scores) = good_and_bad_scores
        goodness_weighted_sum = self.explainability_of_features * self.importance_of_explainability \
                                + goodness_scores * self.importance_of_fitness

        badness_weighted_sum = self.explainability_of_features * self.importance_of_explainability \
                               + badness_scores * self.importance_of_fitness
        return (goodness_weighted_sum, badness_weighted_sum)

    def get_explainable_features(self, criteria='all'):
        amount_to_keep = self.search_space.total_cardinality

        def get_best_using_scores(criteria_scores):
            with_scores = list(zip(self.explainable_features, criteria_scores))
            with_scores.sort(key=utils.second, reverse=True)
            return with_scores[:amount_to_keep]

        if criteria == 'all':
            return self.explainable_features

        scores = None
        if criteria == 'fitness':
            scores = self.get_fitness_unfitness_scores()
        elif criteria == 'popularity':
            scores = self.get_popularity_unpopularity_scores()

        (goodness, badness) = scores
        return (get_best_using_scores(goodness), get_best_using_scores(badness))





