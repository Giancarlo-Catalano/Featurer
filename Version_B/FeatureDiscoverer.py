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

    def __init__(self, search_space, candidateC_population, fitness_scores, merging_power, complexity_function, complexity_damping = 2,
                 importance_of_explainability=0.5):
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.candidate_matrix = self.hot_encoder.to_hot_encoded_matrix(candidateC_population)
        self.fitness_scores = np.array(fitness_scores)
        self.merging_power = merging_power
        self.complexity_function = complexity_function
        self.importance_of_explainability = importance_of_explainability
        self.complexity_damping = complexity_damping



    @property
    def importance_of_fitness(self):
        return 1.0-self.importance_of_explainability

    def get_initial_features(self):
        """returns the trivial features to start from """
        trivial_candidatesC = self.search_space.get_all_trivial_features()
        return [self.hot_encoder.to_hot_encoding(candidateC) for candidateC in trivial_candidatesC]

    def get_flat_feature_clash_matrix(self, trivialH_pool):
        """returns a matrix where element i j is a boolean indicating
           whether the features i and j can coexist in the same individual
           (for implementation reasons, 0 = yes, 1 = no, unfortunately
        """

        """TODO: currently it does n^2 checks, but it could do roughly half of them
                because can_merge is a commutative relation"""

        def features_clash(featureH_a, featureH_b):
            merged_feature = HotEncoding.merge_features(featureH_a, featureH_b)
            is_invalid = not HotEncoding.feature_is_valid_in_search_space(merged_feature, self.search_space)
            return is_invalid

        amount_of_features = len(trivialH_pool)
        clash_matrix = np.zeros((amount_of_features, amount_of_features), dtype=np.float)
        for (index_row, feature_row) in enumerate(trivialH_pool):
            for (index_column, feature_column) in enumerate(trivialH_pool):
                clash_matrix[index_row, index_column] += float(features_clash(feature_row, feature_column))

        return clash_matrix.ravel()

    def fast_is_feature_valid(self, featureH, flat_feature_clash_matrix):
        """returns true if the feature is valid in the search space, by using a matrix multiplication"""
        return utils.flat_outer_product(featureH) @ flat_feature_clash_matrix == 0


    def get_explainability_of_feature(self, featureH):
        """ returns a score in [0,1] describing how explainable the feature is, based on the given complexity function"""
        featureC = self.hot_encoder.feature_from_hot_encoding(featureH)
        raw_complexity = self.complexity_function(featureC)
        dampened_complexity = (1.0/self.complexity_damping)*(raw_complexity-0.5)+0.5

        return 1.0-dampened_complexity

    def fast_features_are_valid(self, featureH_matrix, flat_feature_clash_matrix):
        flat_outer_for_each_feature = utils.row_wise_self_outer_product(featureH_matrix)
        return (flat_outer_for_each_feature @ flat_feature_clash_matrix)

    def get_valid_mergings_of_features(self, featureH_pool, trivialH_pool, at_most):
        """returns *all* the *valid* merges obtained by choosing at_most features from feature_pool"""
        organised_by_weight = list(list() for _ in range(at_most + 1))
        organised_by_weight[0].append(self.hot_encoder.empty_feature)

        flat_clash_matrix = self.get_flat_feature_clash_matrix(trivialH_pool)

        def new_merges_when_adding_feature_to_weight_category(new_feature, weight_category):
            if len(organised_by_weight[weight_category]) == 0:
                return []

            result = []
            all_merging_candidates = np.array([HotEncoding.merge_features(old_feature, new_feature)
                                               for old_feature in organised_by_weight[weight_category]])
            clash_results = self.fast_features_are_valid(all_merging_candidates, flat_clash_matrix)
            for (merged, clashes) in zip(all_merging_candidates, clash_results):
                if not clashes:
                    result.append(merged)
            return result


        def select_explainable_features(featureH_list):
            """given a list of features, it returns the ones that it thinks are worth keeping, based on their explainabilty"""
            """ Effectively, it does a truncation selection of the approximate complexity"""
            if len(featureH_list) < self.search_space.total_cardinality:
                # print(f"A list of {len(featureH_list)} was spared")
                return featureH_list

            # deb_initial_size = len(featureH_list)

            complexity_scores = np.array([self.complexity_function(self.hot_encoder.feature_from_hot_encoding(featureH))
                               for featureH in featureH_list])

            # we then select the top importance_of_explainability of the population, scored by explainability
            threshold = utils.arithmetic_weighted_average(np.min(complexity_scores), self.importance_of_explainability,
                                                          np.max(complexity_scores), self.importance_of_fitness)


            # deb_final_size = len([featureH for (featureH, complexity) in zip(featureH_list, complexity_scores)
            #                  if complexity <= threshold])
            # print(f"The list was filtered from {deb_initial_size} to {deb_final_size}")
            return [featureH for (featureH, complexity) in zip(featureH_list, complexity_scores)
                             if complexity <= threshold]


        def aggressively_select_explainable_features(featureH_list):
            """given a list of features, it returns the ones that it thinks are worth keeping, based on their explainabilty"""
            """ Effectively, it does a truncation selection of the approximate complexity"""

            if len(featureH_list) < self.search_space.total_cardinality:
                # print(f"A list of {len(featureH_list)} was spared")
                return featureH_list


            ideal_size = self.search_space.dimensions**2
            current_size = len(featureH_list)


            # deb_initial_size = len(featureH_list)

            complexity_scores = np.array([self.complexity_function(self.hot_encoder.feature_from_hot_encoding(featureH))
                               for featureH in featureH_list])

            # we then select the top importance_of_explainability of the population, scored by explainability
            portion_to_keep = (current_size/ideal_size)**2
            threshold = utils.arithmetic_weighted_average(np.min(complexity_scores), 1-portion_to_keep,
                                                          np.max(complexity_scores), portion_to_keep)


            # deb_final_size = len([featureH for (featureH, complexity) in zip(featureH_list, complexity_scores)
            #                  if complexity <= threshold])
            # print(f"The list was filtered from {deb_initial_size} to {deb_final_size}")
            return [featureH for (featureH, complexity) in zip(featureH_list, complexity_scores)
                             if complexity <= threshold]

        def consider_feature(feature):
            for weight in reversed(range(at_most)):  # NOTE: the lack of +1
                organised_by_weight[weight + 1].extend(
                    aggressively_select_explainable_features(new_merges_when_adding_feature_to_weight_category(feature, weight)))

        for feature in featureH_pool:
            consider_feature(feature)

        return utils.concat(
            organised_by_weight[1:])  # TODO remove duplicates, unfortunately np.ndarray in not hashable...

    def which_candidates_contain_which_features(self, featureH_pool):
        """returns a matrix in which element i, j is 1 if candidate i contains feature j"""
        feature_matrix = np.transpose(np.array(featureH_pool))
        positive_when_absent = (1 - self.candidate_matrix) @ feature_matrix
        return 1 - np.minimum(positive_when_absent, 1)


    def get_percentile_of_average_fitness_of_features(self, featureH_pool):
        """returns an array with the scores of the given features, adjusted by complexity and statistical significance"""

        feature_presence_matrix = self.which_candidates_contain_which_features(featureH_pool)
        count_for_each_feature = np.sum(feature_presence_matrix, axis=0)
        sum_of_fitness_for_each_feature = np.sum(feature_presence_matrix * utils.to_column_vector(self.fitness_scores),
                                                 axis=0)

        average_fitnesses = np.array([total / float(count) if count > 0 else 0.0 for total, count in
                                      zip(sum_of_fitness_for_each_feature, count_for_each_feature)])

        # then we remap the fitnesses to be between 0 and 1
        # knowing that all the fitnesses are between a min and a max, we will have min become 0 and max become 1
        # effectively applying remap(fitness, from=(min, max), to=(0, 1))

        min_fitness = np.min(self.fitness_scores)
        max_fitness = np.max(self.fitness_scores)

        if (min_fitness == max_fitness):
            return average_fitnesses / min_fitness # should be all ones! TODO perhaps these should be all 0.5?

        unboosted = (average_fitnesses - min_fitness)/(max_fitness-min_fitness)  # forces them to be between 0 and 1

        def boost(x):
            return 3*x**2-2*x**3  # this is the solution to integral([x*(1-x)]^k) for k=1

        return boost(unboosted)

    def get_explainability_of_features(self, featureH_pool):
        """returns an array with the respective explainability score of each feature"""
        return np.array([self.get_explainability_of_feature(featureH) for featureH in featureH_pool])


    def get_approximate_complexity_of_feature(self, featureH):
        """ During feature search, it's necessary to only consider the features that are explainable"""
        """ This is because there are simply too many features, and we exclude in advance the least explainable ones"""
        """ While this could be done considering the fitness as well, it's not predictable how mergings affect fitness"""
        """ Whereas we can assume that complexity worsens through mergings """
        """ Since calculating the complexity can be slow, we use a surrogate: the weight of the feature"""
        return np.sum(featureH)

    def get_scores_of_features(self, featureH_pool):
        """returns the scores for each feature, which consider both the average_fitness and the """
        fitness_percentiles = self.get_percentile_of_average_fitness_of_features(featureH_pool)
        explainability_scores = self.get_explainability_of_features(featureH_pool)
        weighted_sum = explainability_scores * self.importance_of_explainability \
                       + fitness_percentiles * self.importance_of_fitness

        # DEBUG
        """
        print("The scores of all the features are:")
        for (feature_H, explainability, fitness) in zip(featureH_pool, explainability_scores, fitness_percentiles):
            featureC = self.hot_encoder.feature_from_hot_encoding(feature_H)
            e = explainability
            f = fitness
            w_e = self.importance_of_explainability
            w_f = 1-self.importance_of_explainability
            arithmetic = utils.arithmetic_weighted_average(e, w_e, f, w_f)
            geometric = utils.geometric_weighted_average(e, w_e, f, w_f)
            harmonic = utils.harmonic_weighted_average(e, w_e, f, w_f)
            print(f"{featureC} has explainability = {explainability:.2f}, fitness = {fitness:.2f}, "
                  f"ar_m = {arithmetic:.2f},"
                  f"ge_m = {geometric:.2f},"
                  f"ha_m = {harmonic:.2f}")
        """
        return weighted_sum

    def get_next_wave_of_features(self, current_featureH_pool, trivial_features):
        """returns the best mergings of the given pool of features"""
        valid_mergings = self.get_valid_mergings_of_features(current_featureH_pool, trivial_features,
                                                             at_most=self.merging_power)
        scores = self.get_scores_of_features(valid_mergings)

        sorted_by_score = sorted(zip(valid_mergings, scores), key=utils.second, reverse=True)
        amount_to_keep = self.search_space.total_cardinality**2  # freshly pulled out of my arse!
        # return [merged_feature for (merged_feature, _) in sorted_by_score[:amount_to_keep]]
        return sorted_by_score[:amount_to_keep]
