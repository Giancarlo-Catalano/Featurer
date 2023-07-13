import itertools

import CooccurrenceModel
import HotEncoding
import SearchSpace
import utils
import numpy as np
import math


class SimplerInducedClustering:
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

    def build_cooc_model(self, new_features, fitness_list):
        self.cooccurrence_model = CooccurrenceModel.CooccurrenceModel(self.search_space,
                                                                      new_features,
                                                                      self.candidate_matrix,
                                                                      fitness_list)

    def __init__(self, combinatorial_candidates,
                 fitness_list,
                 search_space: SearchSpace.SearchSpace,
                 feature_complexity_evaluator,
                 importance_of_explainability=0.5):

        # set own attributes
        self.search_space = search_space
        hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.candidate_matrix = hot_encoder.to_hot_encoded_matrix(combinatorial_candidates)
        self.fitness_list = fitness_list
        self.feature_complexity_evaluator = feature_complexity_evaluator
        self.importance_of_explainability = importance_of_explainability

        # construct the co-occ model
        trivial_features = hot_encoder.get_hot_encoded_trivial_features()
        self.build_cooc_model(trivial_features, self.fitness_list)

        # the initial features are just the trivial ones
        def adjust_from_diagonal_value(feature, diagonal_value):
            normalised_in_cooc = self.cooccurrence_model.normalise_score_of_feature_vector(feature, diagonal_value)
            return self.get_adjusted_score(feature, normalised_in_cooc)
        trivial_features_scores = [adjust_from_diagonal_value(feature, diag_value)
                                   for (feature, diag_value)
                                   in zip(trivial_features, self.cooccurrence_model.diagonals)]
        self.pool_of_features = list(zip(trivial_features, trivial_features_scores))

    @staticmethod
    def significance_of_observation(observed, expected):
        ratio = observed/expected  # expected should never be 0 !
        spread = 2
        return utils.sigmoid(spread*(ratio-1))
        #return 1-(1/(ratio+1))

    def get_adjusted_score(self, feature_raw, observed_prop):
        feature_combinatorial = self.feature_from_hot_encoded(feature_raw)
        explainability = 1 - self.feature_complexity_evaluator(feature_combinatorial)
        expected_prop = self.search_space.probability_of_feature_in_uniform(feature_combinatorial)
        significance = self.significance_of_observation(observed_prop, expected_prop)
        alpha = self.importance_of_explainability

        weighted_sum = alpha * explainability + (1 - alpha) * significance  # a weighted sum on alpha

        print(
            f"Score of {feature_combinatorial} has "
            f"expl={explainability:.2f}, signif({observed_prop:.2f}|{expected_prop:.2f})={significance:.2f}")
        return weighted_sum

    def get_adjusted_scores(self, features_raw, old_scores):
        return [self.get_adjusted_score(feature, old_score) for (feature, old_score) in zip(features_raw, old_scores)]

    @staticmethod
    def produce_feature_combination_pairs(amount_of_old, amount_of_new):
        old_indexes = range(0, amount_of_old)
        new_indexes = range(amount_of_old, amount_of_old+amount_of_new)
        old_with_new = [(from_old, from_new) for from_old in old_indexes for from_new in new_indexes]
        new_with_new = list(itertools.combinations(new_indexes, 2))  # prevents (a, b) and (b, a), and also (a, a)
        return old_with_new+new_with_new

    @property
    def amount_of_features_in_model(self):
        return self.cooccurrence_model.feature_group.amount_of_features

    def amount_of_features_in_pool(self):
        return len(self.pool_of_features)

    @property
    def trivial_features(self):
        return self.cooccurrence_model.feature_group.hot_features

    def pair_as_feature_vector(self, pair):
        result = np.zeros(self.amount_of_features_in_model, dtype=utils.float_type)
        result[pair[0]] = 1.0
        result[pair[1]] = 1.0
        return result

    def pair_as_raw_feature(self, pair):
        def get_nth_raw_feature_from_pool(n):
            return self.pool_of_features[n][0]
        return HotEncoding.merge_features(get_nth_raw_feature_from_pool(pair[0]),
                                          get_nth_raw_feature_from_pool(pair[1]))

    def get_tentative_new_features(self, amount_of_old, amount_of_new):
        """returns a list of possible features that should be added to the list of features"""
        pairs_to_attempt_to_merge = self.produce_feature_combination_pairs(amount_of_old, amount_of_new)
        raw_forms = [self.pair_as_raw_feature(pair) for pair in pairs_to_attempt_to_merge]

        return raw_forms

    def get_features_to_add(self, amount_of_old, amount_of_new):
        """Generates the new features to add, along with their scores"""
        # generate all the possible merges, even some invalid ones
        proposal = self.get_tentative_new_features(amount_of_old, amount_of_new)

        # remove invalid merges, where the hot encoding for a variable has too many ones
        # also remove features which have been added already
        proposal = [raw_form for raw_form in proposal
                    if self.is_valid_hot_encoded_feature(raw_form)
                    if not self.feature_is_present_already(raw_form)]

        # remove duplicate entries, which are unavoidable
        proposal = utils.remove_duplicates_unhashable(proposal, custom_equals=np.array_equal)

        # first, get the score according to the cooccurrence model, and then take into consideration the explainability
        scores = [self.cooccurrence_model.score_of_feature_vector(raw_form) for raw_form in proposal]
        adjusted_scores = self.get_adjusted_scores(proposal, scores)

        # lastly, keep the features which have the highest adjusted score
        amount_to_keep = self.search_space.total_cardinality  # IMPORTANT
        result = list(zip(proposal, adjusted_scores))
        result.sort(key=utils.second, reverse=True)
        return result[:amount_to_keep]

    def is_valid_hot_encoded_feature(self, feature):
        """returns true when the feature is valid (would have only numbers and None as its values)"""
        return self.cooccurrence_model.feature_group.hot_encoder.is_valid_hot_encoded_feature(feature)

    def feature_is_present_already(self, raw_feature):
        # TODO make this more efficient
        for present_feature in self.pool_of_features:
            if np.array_equal(present_feature, raw_feature):
                return True
        return False

    def expand_features_iteratively(self, amount_of_iterations):
        print(f"Start of expand_features_iteratively({amount_of_iterations})")
        amount_of_old = 0
        amount_of_new = self.amount_of_features_in_pool()
        for iteration in range(0, amount_of_iterations):
            print(f"Start of iteration #{iteration}")
            print(f"At this point, the cooc matrix is \n{self.cooccurrence_model.cooccurrence_matrix}")
            print(f"(and the maximum value of the cooc matrix is {self.cooccurrence_model.maximum_value})")
            new_list_of_features = self.get_features_to_add(amount_of_old, amount_of_new)

            print(f"The proposed expansion is ")
            self.pretty_print_list_of_features(utils.unzip(new_list_of_features)[0])

            self.pool_of_features.extend(new_list_of_features)
            amount_of_old = amount_of_old+amount_of_new
            amount_of_new = len(new_list_of_features)
            print(f"amount_of_old = {amount_of_old}, "
                  f"amount_of_new = {amount_of_new}, ")

    def get_most_relevant_features(self):
        result = [(self.feature_from_hot_encoded(raw_feature), score) for (raw_feature, score) in self.pool_of_features]
        return utils.remove_duplicates_unhashable(result, key=utils.second, custom_equals=np.array_equal)

    def build(self):
        amount_of_iterations = math.ceil(math.log2(self.search_space.amount_of_trivial_features))
        self.expand_features_iteratively(amount_of_iterations)

    def feature_from_hot_encoded(self, hot_encoded_feature):
        return self.cooccurrence_model.feature_group.hot_encoder.feature_from_hot_encoding(hot_encoded_feature)

    def pretty_print_list_of_features(self, hot_encoded_features):
        combinatorial_features = [self.feature_from_hot_encoded(hot) for hot in hot_encoded_features]
        for comb_feature in combinatorial_features:
            print(f"\t {comb_feature}")

    def __repr__(self):
        result = f"candidate matrix = \n{self.candidate_matrix}\n"
        result += f"feature list = \n"

        for (hot_feature, score) in self.pool_of_features:
            feature_combinatorial = self.feature_from_hot_encoded(hot_feature)
            result += f"\t{feature_combinatorial.__repr__()}, score={score}"

        result += "\n"

        result += f"The coocc matrix = \n{self.cooccurrence_model.cooccurrence_matrix}\n"
        return result
