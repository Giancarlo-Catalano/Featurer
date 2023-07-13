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

    def build_cooc_model(self, new_features):
        self.cooccurrence_model = CooccurrenceModel.CooccurrenceModel(self.search_space,
                                                                      new_features,
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
        """the size of an initial feature is ceil(clique/2)"""
        weight_of_initial_feature = self.get_initial_feature_weight()
        merging_instructions = itertools.combinations(range(self.search_space.total_cardinality), weight_of_initial_feature)

        def raw_feature_ex_nihilo(merge_list):
            result = np.zeros(self.search_space.amount_of_trivial_features, dtype=utils.float_type)
            for which in merge_list:
                result[which] = 1.0
            return result

        raw_features = [raw_feature_ex_nihilo(merge_list) for merge_list in merging_instructions]
        filtered = [feature for feature in raw_features if self.is_valid_hot_encoded_feature(feature)]

        return filtered

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
        initial_features = self.get_initial_features()
        #print(f"The initial_features are")
        #self.pretty_print_features(initial_features)

        self.build_cooc_model(initial_features)
        self.max_weight_of_interaction_so_far = self.get_initial_feature_weight()

        # set up the starting cooccurrence model
        self.pool_of_features = initial_features
        self.present_combinatorial_features = set([self.hot_encoder.feature_from_hot_encoding(f) for f in self.pool_of_features])

    def get_adjusted_score(self, feature_raw, observed_prop): #TODO replace feature raw with feature_combinatorial
        def significance_of_observation(observed, expected):
            ratio = observed / expected  # expected should never be 0 !
            spread = 2
            return utils.sigmoid(spread * (ratio - 1 - (1 / spread)))
            # return 1-(1/(ratio+1))

        def experimental_significance_of_observation(observed, expected):
            if observed < expected:
                return 0.0
            chi_squared = ((observed - expected) ** 2) / expected

            def normalise(x):
                return 1 - math.exp(-1 * x)

            return normalise(chi_squared)

        feature_combinatorial = self.feature_from_hot_encoded(feature_raw)
        explainability = 1 - self.feature_complexity_evaluator(feature_combinatorial)
        expected_prop = self.search_space.probability_of_feature_in_uniform(feature_combinatorial)
        significance = significance_of_observation(observed_prop, expected_prop)
        alpha = self.importance_of_explainability

        weighted_sum = alpha * explainability + (1 - alpha) * significance  # a weighted sum on alpha

        print(
            f"Score of {feature_combinatorial} has "
            f"expl={explainability:.2f}, signif({observed_prop:.2f}|{expected_prop:.2f})={significance:.2f}")
        return weighted_sum

    def get_adjusted_scores(self, features_raw, old_scores):
        return [self.get_adjusted_score(feature, old_score) for (feature, old_score) in zip(features_raw, old_scores)]

    @property
    def amount_of_features_in_model(self):
        return self.cooccurrence_model.feature_group.amount_of_features

    @property
    def amount_of_features_in_pool(self):
        return len(self.pool_of_features)

    def merging_instructions_to_feature_vector(self, merging_instructions):
        result = np.zeros(self.amount_of_features_in_model, dtype=utils.float_type)
        for which in merging_instructions:
            result[which] = 1.0
        return result

    def merging_instructions_as_raw_feature(self, merging_instructions):
        return HotEncoding.merge_many_features([self.pool_of_features[n] for n in merging_instructions])

    def score_of_feature(self, feature_vector, feature_raw):
        observed_prop = self.cooccurrence_model.score_of_feature_vector(feature_vector)
        return self.get_adjusted_score(feature_raw, observed_prop)

    def is_valid_hot_encoded_feature(self, feature):
        """returns true when the feature is valid (would have only numbers and None as its values)"""
        return self.hot_encoder.is_valid_hot_encoded_feature(feature)

    def feature_is_present_already(self, raw_feature):
        # TODO make this more efficient
        for present_feature in self.pool_of_features:
            if HotEncoding.features_are_equal(present_feature, raw_feature):
                return True
        return False

    def raw_feature_is_acceptable(self, raw_feature):
        is_new = not self.feature_is_present_already(raw_feature)
        is_valid = (HotEncoding.feature_is_valid_in_search_space(raw_feature, self.search_space))

        return is_new and is_valid

    def get_features_to_be_added(self, clique_size):
        merging_instructions = list(itertools.combinations(range(self.amount_of_features_in_pool), clique_size))
        features_in_model_and_raw = [
            (self.merging_instructions_to_feature_vector(merge_list),
             self.merging_instructions_as_raw_feature(merge_list))
            for merge_list in merging_instructions]

        features_in_model_raw_combinatorial = [(in_model, in_raw, self.hot_encoder.feature_from_hot_encoding(in_raw)) for (in_model, in_raw) in features_in_model_and_raw]

        def combinatorial_is_valid(combinatorial):
            return all([val is None or not math.isnan(val) for val in combinatorial.values])

        features_in_model_raw_combinatorial = [(in_model, raw, combinatorial) for (in_model, raw, combinatorial) in features_in_model_raw_combinatorial
                                     if combinatorial_is_valid(combinatorial)
                                     if combinatorial not in self.present_combinatorial_features]

        features_in_model_raw_combinatorial = utils.remove_duplicates(features_in_model_raw_combinatorial, key = lambda x: x[2])

        raw_feature_and_scores = [(raw, self.score_of_feature(fv, raw)) for (fv, raw, _) in features_in_model_raw_combinatorial]
        raw_feature_and_scores.sort(key=utils.second, reverse=True)
        amount_to_keep = self.search_space.total_cardinality*2
        result = [raw_feature for (raw_feature, score) in raw_feature_and_scores[:amount_to_keep]]
        return result

    def old_get_features_to_be_added(self, clique_size):
        merging_instructions = list(itertools.combinations(range(self.amount_of_features_in_pool), clique_size))
        features_in_model_and_raw = [
            (self.merging_instructions_to_feature_vector(merge_list),
             self.merging_instructions_as_raw_feature(merge_list))
            for merge_list in merging_instructions]


        features_in_model_and_raw = [(in_model, raw) for (in_model, raw) in features_in_model_and_raw
                                     if HotEncoding.feature_is_valid_in_search_space(raw, self.search_space)
                                     if not self.feature_is_present_already(raw)]

        features_in_model_and_raw = utils.remove_duplicates_unhashable(features_in_model_and_raw, key=utils.second,
                                                                       custom_equals=HotEncoding.features_are_equal)

        raw_feature_and_scores = [(raw, self.score_of_feature(fv, raw)) for (fv, raw) in features_in_model_and_raw]
        raw_feature_and_scores.sort(key=utils.second, reverse=True)
        amount_to_keep = self.search_space.total_cardinality * 2
        result = [raw_feature for (raw_feature, score) in raw_feature_and_scores[:amount_to_keep]]
        return result

    def expand_features(self):
        print(f"Called expand_features")
        new_list_of_features = self.get_features_to_be_added(self.merging_power)

        self.pool_of_features.extend(new_list_of_features)

        self.present_combinatorial_features.update([self.hot_encoder.feature_from_hot_encoding(f) for f in new_list_of_features])

        self.build_cooc_model(self.pool_of_features)
        self.max_weight_of_interaction_so_far *= self.merging_power

        print(f"The current pool of features is ")
        self.pretty_print_features(self.pool_of_features)

    def get_most_relevant_features(self):
        scores = self.get_scores_of_features_in_model(adjust_scores_considering_complexity=True)
        result = [(self.feature_from_hot_encoded(raw_feature), score)
                  for (raw_feature, score) in zip(self.pool_of_features, scores)]
        return result

    def build(self):
        while self.max_weight_of_interaction_so_far < math.ceil(self.search_space.dimensions / 2):
            self.expand_features()

    def feature_from_hot_encoded(self, hot_encoded_feature):
        return self.hot_encoder.feature_from_hot_encoding(hot_encoded_feature)

    def pretty_print_features_with_scores(self, features_with_scores):
        for hot, score in features_with_scores:
            print(f"\t {self.feature_from_hot_encoded(hot)} with score {score:.2f}")

    def pretty_print_features(self, features):
        print("The features are:")
        for feature in features:
            print(f"\t {self.feature_from_hot_encoded(feature)}")

    def __repr__(self):
        result = f"candidate matrix = \n{self.candidate_matrix}\n"
        result += f"feature list = \n"

        for (hot_feature) in self.pool_of_features:
            feature_combinatorial = self.feature_from_hot_encoded(hot_feature)
            result += f"\t{feature_combinatorial.__repr__()}"

        result += "\n"

        result += f"The coocc matrix = \n{self.cooccurrence_model.cooccurrence_matrix}\n"
        return result

    def score_of_candidate(self, candidate_combinatorial):
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
            return self.merging_instructions_as_raw_feature(result_feature_indexes)

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
        # print(f"The starting candidate is {current}")
        current = recalculate_feature_indexes(current)

        while not result_is_complete(current):
            # print(f"start of loop, current = {current}")
            distribution_of_features = get_distribution_for_current_result(current)
            # print(f"The distribution of features is {distribution_of_features}")
            potential_new_candidate = None
            while True:
                potential_new_candidate = sample_new_feature_and_add(current, distribution_of_features)
                if result_is_valid(potential_new_candidate):
                    # print(f"found a successfull addition: {potential_new_candidate}")
                    break  # succeed!
            current = potential_new_candidate

        # print(f"Reached the end of MCMC, the result is {current}")
        return self.hot_encoder.candidate_from_hot_encoding(get_raw_form(current))