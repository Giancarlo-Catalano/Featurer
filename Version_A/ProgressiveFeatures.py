import itertools

import CooccurrenceModel
import HotEncoding
import SearchSpace
import utils
import numpy as np
import math


class ProgressiveFeatures:
    """
    attributes:
        hot_encoded_candidates (raw, not features)
        coocc model
        feature_complexity_evaluator  (a value from 0 to 1, higher for more complex (less explainable) clusters)

    parameters
        proportion_of_features_to_keep_per_iteration = search_space.total_cardinality
        importance_of_explainability (from 0 to 1)

    General Explanation:
        * generates a list of features which are "correlated" with the fitnesses of the given candidates
        * These features can be inspected for explainability purposes
        * The cooc_model obtained can also be used as a surrogate fitness function

    Usage:
        * The constructor needs
            * a search space
            * a list of candidates, and a list their fitnesses
            * a function which evaluates the complexity of a feature
            * a parameter from 0 to 1 determining how "simple" the resulting features should be
        * After construction, call .build() to obtain the features
    """
    search_space: SearchSpace.SearchSpace
    cooccurrence_model: CooccurrenceModel.CooccurrenceModel

    def build_cooc_model(self):
        print(f"In build_cooc_model, the features at this point are {self.pool_of_features}")
        raw_hot_encoded_features = [self.hot_encoder.to_hot_encoding(combinatorial_feature)
                                    for combinatorial_feature in self.pool_of_features]

        print(f"The raw encoded features are {raw_hot_encoded_features}")
        self.cooccurrence_model = CooccurrenceModel.CooccurrenceModel(self.search_space,
                                                                      raw_hot_encoded_features,
                                                                      self.candidate_matrix,
                                                                      self.fitness_list)

    def get_scores_of_features_in_model(self, adjust_scores_considering_complexity=True):
        unadjusted = self.cooccurrence_model.get_scores_of_own_features()
        if adjust_scores_considering_complexity:
            return self.get_adjusted_scores(self.pool_of_features, unadjusted)
        else:
            return unadjusted

    def get_initial_feature_weight(self):
        return math.ceil(self.merging_power / 2)

    def get_initial_features(self):
        # TODO this could be more efficient !
        weight_of_initial_feature = self.get_initial_feature_weight()
        trivial_features = self.search_space.get_all_trivial_features()
        if weight_of_initial_feature == 1:
            return trivial_features

        combinations_to_merge = itertools.combinations(trivial_features, weight_of_initial_feature)
        initial_features = []
        for combination in combinations_to_merge:
            merged = self.search_space.merge_features(combination)
            if merged is not None:
                initial_features.append(merged)
        return initial_features

    def __init__(self, combinatorial_candidates,
                 fitness_list,
                 search_space: SearchSpace.SearchSpace,
                 feature_complexity_evaluator,
                 importance_of_explainability=0.5,
                 merging_power=2):

        # set own attributes
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.candidate_matrix = self.hot_encoder.to_hot_encoded_matrix(combinatorial_candidates)
        self.fitness_list = fitness_list
        self.feature_complexity_evaluator = feature_complexity_evaluator
        self.importance_of_explainability = importance_of_explainability
        self.merging_power = merging_power

        # construct the co-occ model
        self.pool_of_features = set(self.get_initial_features())
        print(f"In the constructor of Progressive Features, the pool of features is {self.pool_of_features}")
        self.build_cooc_model()
        self.max_weight_of_interaction_so_far = self.get_initial_feature_weight()

        # set up the starting cooccurrence model

    def get_adjusted_score(self, feature_combinatorial,
                           observed_prop):  # TODO replace feature raw with feature_combinatorial
        def significance_of_observation(observed, expected):
            ratio = observed / expected  # expected should never be 0 !
            spread = 2
            return utils.sigmoid(spread * (ratio - 1 - (1 / spread)))
            # return 1-(1/(ratio+1))

        def experimental_significance_of_observation(observed, expected):
            if observed < expected:
                return 0.0
            chi_squared = utils.chi_squared(observed, expected)
            normalisation_denominator = utils.chi_squared(1.0, expected)

            return chi_squared/normalisation_denominator


        explainability = 1 - self.feature_complexity_evaluator(feature_combinatorial)
        expected_prop = self.search_space.probability_of_feature_in_uniform(feature_combinatorial)
        significance = significance_of_observation(observed_prop, expected_prop)
        alpha = self.importance_of_explainability

        weighted_sum = alpha * explainability + (1 - alpha) * significance  # a weighted sum on alpha

        print(
            f"Score of {feature_combinatorial} has "
            f"expl={explainability:.2f}, signif({observed_prop:.2f}|{expected_prop:.2f})={significance:.2f}")
        return weighted_sum

    def get_adjusted_scores(self, features_combinatorial, old_scores):
        return [self.get_adjusted_score(feature, old_score) for (feature, old_score) in
                zip(features_combinatorial, old_scores)]

    @property
    def amount_of_features_in_pool(self):
        return len(self.pool_of_features)

    def merging_instructions_to_feature_vector(self, merging_instructions):
        result = np.zeros(self.amount_of_features_in_pool, dtype=utils.float_type)
        for which in merging_instructions:
            result[which] = 1.0
        return result

    def score_of_feature(self, feature_vector, feature_combinatorial):
        observed_prop = self.cooccurrence_model.score_of_feature_vector(feature_vector)
        return self.get_adjusted_score(feature_combinatorial, observed_prop)

    def get_features_to_be_added(self, clique_size):
        merging_candidates = []

        for merging_instructions in itertools.combinations(enumerate(self.pool_of_features), clique_size):
            (indices, merge_items) = utils.unzip(merging_instructions)

            merged_combinatorial = self.search_space.merge_features(merge_items)
            if (merged_combinatorial is not None) and (merged_combinatorial not in self.pool_of_features):
                feature_form = self.merging_instructions_to_feature_vector(indices)
                # TODO here I should recalculate which features are present...
                score_of_feature = self.score_of_feature(feature_form, merged_combinatorial)
                merging_candidates.append((merged_combinatorial, score_of_feature))

        merging_candidates.sort(key=utils.second, reverse=True)
        amount_to_keep = self.search_space.total_cardinality * 2
        result = [raw_feature for (raw_feature, score) in merging_candidates[:amount_to_keep]]
        return result

    def expand_features(self):
        print(f"Called expand_features")
        new_list_of_features = self.get_features_to_be_added(self.merging_power)

        self.pool_of_features.update(new_list_of_features)

        self.build_cooc_model()
        self.max_weight_of_interaction_so_far *= self.merging_power

        print(f"The current pool of features is ")
        self.pretty_print_features(self.pool_of_features)

    def get_most_relevant_features(self):
        scores = self.get_scores_of_features_in_model(adjust_scores_considering_complexity=True)
        return list(zip(self.pool_of_features, scores))

    def build(self):
        while self.max_weight_of_interaction_so_far < math.ceil(self.search_space.dimensions / 2):
            self.expand_features()

    def feature_from_hot_encoded(self, hot_encoded_feature):
        return self.hot_encoder.feature_from_hot_encoding(hot_encoded_feature)

    def pretty_print_features_with_scores(self, features_with_scores):
        for combinatorial, score in features_with_scores:
            print(f"\t {combinatorial} with score {score:.2f}")

    def pretty_print_features(self, features):
        print("The features are:")
        for combinatorial_feature in features:
            print(f"\t {combinatorial_feature}")

    def __repr__(self):
        result = f"candidate matrix = \n{self.candidate_matrix}\n"
        result += f"feature list = \n"

        for (feature_combinatorial) in self.pool_of_features:
            result += f"\t{feature_combinatorial.__repr__()}"
        result += "\n"
        result += f"The coocc matrix = \n{self.cooccurrence_model.cooccurrence_matrix}\n"
        return result

    def score_of_candidate(self, candidate_combinatorial):
        """This is to be used on candidates generated from the outside"""
        candidate_hot_encoded = self.hot_encoder.to_hot_encoding(candidate_combinatorial)
        score_according_to_feature_group = self.cooccurrence_model.score_of_raw_candidate_vector(candidate_hot_encoded)
        return score_according_to_feature_group

    def MCMC(self):
        """Generates a new candidate using Markov Chain Monte Carlo"""

        def get_starting_candidate():
            distribution_grid = np.ndarray.tolist(self.cooccurrence_model.cooccurrence_matrix)
            result_feature_indexes = list(utils.sample_from_grid_of_weights(distribution_grid))
            return result_feature_indexes

        def get_feature_vector(result_feature_indexes):
            return self.merging_instructions_to_feature_vector(result_feature_indexes)

        def get_raw_form(result_feature_indexes):
            raw_features = [self.cooccurrence_model.feature_group.hot_features[index]
                            for index in result_feature_indexes]
            return HotEncoding.merge_many_features(raw_features)

        def recalculate_feature_indexes(indexes):
            raw_form = get_raw_form(indexes)
            result_as_feature_vector = self.cooccurrence_model.feature_group.get_feature_vector_for_candidate(raw_form)
            recalculated_indexes = [index for (index, value) in enumerate(result_as_feature_vector) if value > 0.5]
            return recalculated_indexes

        def get_distribution_for_current_result(indexes):
            feature_vector = get_feature_vector(indexes)
            distribution = np.ndarray.tolist(feature_vector @ self.cooccurrence_model.cooccurrence_matrix)
            for which in indexes:  # we clear out the features that are already present
                distribution[which] = 0
            return distribution

        def sample_new_feature_and_add(indexes, distribution):
            new_feature_index = utils.sample_index_with_weights(distribution)
            new_indexes = indexes + [new_feature_index]
            return recalculate_feature_indexes(new_indexes)

        def result_is_complete(indexes):
            combinatorial_candidate = self.hot_encoder.candidate_from_hot_encoding(get_raw_form(indexes))
            amount_of_nones = sum([1 if val is None else 0 for val in combinatorial_candidate.values])
            return amount_of_nones == 0

        def result_is_valid(indexes):
            combinatorial_candidate = self.hot_encoder.candidate_from_hot_encoding(get_raw_form(indexes))
            amount_of_nans = sum([0 if (val is None or not math.isnan(val)) else 1
                                  for val in combinatorial_candidate.values])
            return amount_of_nans == 0

        # start of algorithm
        current = get_starting_candidate()
        current = recalculate_feature_indexes(current)

        while not result_is_complete(current):
            distribution_of_features = get_distribution_for_current_result(current)
            potential_new_candidate = None
            while True:
                potential_new_candidate = sample_new_feature_and_add(current, distribution_of_features)
                if result_is_valid(potential_new_candidate):
                    break  # succeed!
            current = potential_new_candidate

        return self.hot_encoder.candidate_from_hot_encoding(get_raw_form(current))
