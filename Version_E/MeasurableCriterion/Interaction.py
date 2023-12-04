import itertools
from typing import Iterable

import numpy as np

import SearchSpace
import utils
from Version_E import HotEncoding
from Version_E.Feature import Feature
from Version_E.MeasurableCriterion.CriterionUtilities import PPICachedData
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.MeasurableCriterion.Robustness import get_fuzzy_match_matrix, get_mean_of_fuzzy_match_matrix
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation

FlatArray = np.ndarray
Matrix = np.ndarray
ColumnVector = np.ndarray

ValuePosition = int  # the index
Fitness = float

InteractionScore = float
InteractionTable = Matrix

HotEncodedFeature = FlatArray


def flatten_var_val(var: int, val: int, search_space: SearchSpace.SearchSpace) -> int:
    return search_space.precomputed_offsets[var] + val


def get_pairs_to_check(search_space: SearchSpace.SearchSpace) -> list[((int, int), (int, int))]:
    amount_of_vars = search_space.dimensions

    different_vars = [(flatten_var_val(var_a, val_a, search_space), flatten_var_val(var_b, val_b, search_space))
                      for var_a in range(amount_of_vars)
                      for val_a in range(search_space.cardinalities[var_a])
                      for var_b in range(var_a + 1, amount_of_vars)  # different vars guaranteed
                      for val_b in range(search_space.cardinalities[var_a])]

    same_var_same_value = [(pos, pos) for pos in range(search_space.total_cardinality)]

    return different_vars + same_var_same_value


def get_normalised_fitnesses(ppi: PrecomputedPopulationInformation) -> FlatArray:
    in_zero_one = utils.remap_array_in_zero_one(ppi.fitness_array)
    with_sum_one = in_zero_one / sum(in_zero_one)
    return with_sum_one


def get_just_value(var: int, val: int, search_space: SearchSpace.SearchSpace) -> HotEncodedFeature:
    just_that_value = np.zeros(search_space.total_cardinality, dtype=float)
    just_that_value[flatten_var_val(var, val, search_space)] = 1.0

    return just_that_value


# FBB is F1X, FX1, F11...I wanted to call it FXX but that makes it really confusing
FBB = (HotEncodedFeature, HotEncodedFeature, HotEncodedFeature)

Var_Val = (int, int)


def get_all_FBB(pairs_to_generate,  # but it's Iterable[(int, int)]
                search_space: SearchSpace.SearchSpace) -> (list[HotEncodedFeature], list[HotEncodedFeature]):
    """ for each value in the search space, we generate the FBB features,
    which are hot encoded and placed in successive rows., HotEncodedFeature)
    Ie, the first four rows are f00..f11 for the first pair of values, and so on"""

    values_as_features = list(np.identity(search_space.total_cardinality, dtype=float))

    # values_as_features[var][val] to access (var, val) as a feature
    def get_F11_for_val_pair(pos_a: int, pos_b: int) -> HotEncodedFeature:
        f1X = values_as_features[pos_a]
        fX1 = values_as_features[pos_b]

        f11 = np.maximum(f1X, fX1)  # is the feature with both

        return f11

    composite_fbbs = [get_F11_for_val_pair(pos_a, pos_b) for pos_a, pos_b in pairs_to_generate]
    return values_as_features, composite_fbbs


def get_pseudo_proportions_for_hot_encoded_features(hot_encoded_features: Iterable[HotEncodedFeature],
                                                    ppi: PrecomputedPopulationInformation) -> np.ndarray:
    # we'll want to get some stats out of these features, so we create a pfi
    pfi = PrecomputedFeatureInformation.get_from_hot_encoded_features(ppi, np.array(hot_encoded_features))

    sum_of_fitnesses = utils.weighted_sum_of_rows(pfi.feature_presence_matrix, get_normalised_fitnesses(ppi))

    return sum_of_fitnesses


def joint_entropy(p1X: float, pX1: float, p11: float) -> float:
    pXX = 1
    p01 = pX1 - p11
    p10 = p1X - p11
    p00 = pXX - p1X - p01

    def entropy(pxy: float) -> float:
        if pxy <= 1e-6:
            return 0.0
        return -pxy * np.log2(pxy)

    return entropy(p00) + entropy(p01) + entropy(p10) + entropy(p11)


def variation_of_information(p1X: float, pX1: float, p11: float) -> float:
    return p11 * (np.log2(p11 / p1X) + np.log(p11 / pX1))


def variation_of_information_2(p1X: float, pX1: float, p11: float) -> float:
    return p1X + pX1 - 2 * joint_entropy(p1X, pX1, p11)


def conditional_entropy(p1X: float, pX1: float, p11: float) -> float:
    pXX = 1
    p01 = pX1 - p11
    p10 = p1X - p11
    p00 = pXX - p1X - p01
    p0X = p00 + p01
    pX0 = p00 + p10

    def aux_mutual_information(pAB, pAX, pXB) -> float:
        """https://en.wikipedia.org/wiki/Mutual_information"""
        denominator = pAX * pXB
        if pAB < 1e-6:
            return 0  # panic
        return pAB * np.log2(pAB / denominator)

    mutual_information = aux_mutual_information(p00, p0X, pX0) + \
                         aux_mutual_information(p01, p0X, pX1) + \
                         aux_mutual_information(p10, p1X, pX0) + \
                         aux_mutual_information(p11, p1X, pX1)

    return mutual_information


def mutual_information(p1X: float, pX1: float, p11: float) -> float:
    denominator = p1X * pX1
    if denominator < 1e-6:
        return 0  # panic
    if p11 < 1e-6:
        return 0  # panic, a bit less. This makes sense!

    mi = p11 * np.log(p11 / denominator)

    return mi


def get_interaction_table(ppi: PrecomputedPopulationInformation) -> Matrix:
    print("Starting to generate the table")
    pairs_to_calculate = get_pairs_to_check(ppi.search_space)

    # note that of all the interactions, we only keep the upper triangle, without even the diagonal, and only between different variables
    amount_of_vals = ppi.search_space.total_cardinality

    marginal_fbbs, composite_fbb = get_all_FBB(pairs_to_calculate, ppi.search_space)
    pseudo_proportions = get_pseudo_proportions_for_hot_encoded_features(marginal_fbbs + composite_fbb, ppi)

    marginal_proportions = pseudo_proportions[:len(marginal_fbbs)]
    composite_proportions = pseudo_proportions[len(marginal_fbbs):]
    table = np.zeros((amount_of_vals, amount_of_vals))

    for a_b, p11 in zip(pairs_to_calculate, composite_proportions):
        position_a, position_b = a_b

        p1X = marginal_proportions[position_a]
        pX1 = marginal_proportions[position_b]

        value_to_put_in_table = mutual_information(p1X=p1X, pX1=pX1, p11=p11)  # might divide it by the joint entropy...
        table[position_a][position_b] = value_to_put_in_table

    print("The table has been generated")
    return table


def get_interactions_for_feature(feature: HotEncodedFeature,
                                 interaction_table: InteractionTable) -> FlatArray:
    boolean_feature = np.array(feature, dtype=bool)
    present_pairs = np.outer(boolean_feature, boolean_feature)
    present_pairs = np.triu(present_pairs)

    return interaction_table[present_pairs]  # this might be wrong


def get_interaction_score_for_feature(feature: HotEncodedFeature,
                                      interaction_table: InteractionTable) -> InteractionScore:
    return np.min(get_interactions_for_feature(feature, interaction_table))


class Interaction(MeasurableCriterion):
    cached_interaction_table: PPICachedData

    def __init__(self):
        self.cached_interaction_table = PPICachedData(get_interaction_table)

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        interaction_table: InteractionTable = self.cached_interaction_table.get_data_for_ppi(
            pfi.precomputed_population_information)

        features_as_rows = pfi.feature_matrix.T

        def score_for_feature(feature: HotEncodedFeature) -> InteractionScore:
            return get_interaction_score_for_feature(feature, interaction_table)

        interaction_scores = np.array(list(map(score_for_feature, features_as_rows)))
        return interaction_scores

    def describe_score(self, given_score) -> str:
        return f"Has interaction score of {given_score}"


def calculate_variance_of_features(pfi: PrecomputedFeatureInformation):
    return np.square(pfi.sd_for_each_feature)


def get_divorce_score(feature: Feature,
                      feature_variance: float,
                      pfi: PrecomputedFeatureInformation) -> float:
    def only_that_var(var: int) -> Feature:
        value = feature.values_mask[var]
        return Feature.from_trivial_feature(var, value, pfi.search_space)

    def without_that_var(var: int) -> Feature:
        return feature.with_value(var, None)

    if feature_variance == 0:
        return 0  # if there is no variation in the feature itself

    vars_in_use = feature.get_used_vars()
    if len(vars_in_use) == 0:
        return 0  # if there are no variables set, what should be the default score?

    features_without_var = list(map(without_that_var, vars_in_use))
    just_vars = list(map(only_that_var, vars_in_use))

    temp_pfi = PrecomputedFeatureInformation(pfi.precomputed_population_information, features_without_var)

    minimum_amount_of_samples = 6
    pairs_to_evaluate = [(without, only)
                         for without, only, count_of_without in
                         zip(features_without_var, just_vars, temp_pfi.count_for_each_feature)
                         if count_of_without > minimum_amount_of_samples]

    if len(pairs_to_evaluate) == 0:
        return 0

    features_to_evaluate = utils.concat_lists(utils.unzip(pairs_to_evaluate))

    evaluation_pfi = PrecomputedFeatureInformation(pfi.precomputed_population_information, features_to_evaluate)

    evaluation_variances = np.square(evaluation_pfi.sd_for_each_feature)
    evaluation_variances = evaluation_variances.reshape((2, len(pairs_to_evaluate)))
    evaluation_variances = np.sum(evaluation_variances, axis=0)

    diff_between_separate_and_together = np.abs(evaluation_variances - feature_variance)

    return np.average(diff_between_separate_and_together)  # maybe it could be the average


class Divorce(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return f"Divorce"

    def get_raw_score_array_old(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        variances = np.square(pfi.sd_for_each_feature)

        divorce_scores = np.array([get_divorce_score(feature, variance, pfi)
                                   for feature, variance in zip(pfi.features, variances)])

        return divorce_scores

    def describe_score(self, given_score) -> str:
        return f"Divorce = {given_score}"


class WeakestLink(MeasurableCriterion):
    cached_normalised_fitnesses: PPICachedData
    cached_pX1s: PPICachedData

    def get_proportions_of_features(self, hot_encoded_features: Iterable[HotEncodedFeature],
                                    ppi: PrecomputedPopulationInformation) -> FlatArray:
        normalised_fitnesses: FlatArray = self.cached_normalised_fitnesses.get_data_for_ppi(ppi)
        temp_pfi = PrecomputedFeatureInformation.get_from_hot_encoded_features(ppi, np.array(hot_encoded_features))
        proportions = utils.weighted_sum_of_rows(temp_pfi.feature_presence_matrix, normalised_fitnesses)
        return proportions

    def __init__(self):
        self.cached_normalised_fitnesses = PPICachedData(get_normalised_fitnesses)

        def get_P1Xs(ppi: PrecomputedPopulationInformation) -> FlatArray:
            amount_of_values = ppi.search_space.total_cardinality
            trivial_features = np.identity(amount_of_values, dtype=float)
            return self.get_proportions_of_features(trivial_features, ppi)

        self.cached_pX1s = PPICachedData(get_P1Xs)

    def linkage_scores_for_feature(self, feature: HotEncodedFeature,
                                   ppi: PrecomputedPopulationInformation,
                                   p11: float) -> list[float]:
        present_vals = [val_pos for val_pos, is_used in enumerate(feature) if is_used]

        if len(present_vals) == 0:
            return []

        def without_that_val(val_pos: int) -> HotEncodedFeature:
            without = np.array(feature)
            without[val_pos] = 0.0
            return without

        f1Xs = list(map(without_that_val, present_vals))

        p1Xs: FlatArray = self.get_proportions_of_features(f1Xs, ppi)
        all_pX1s: FlatArray = self.cached_pX1s.get_data_for_ppi(ppi)
        pX1s = [pX1 for pX1, is_used in zip(all_pX1s, feature) if is_used]

        result = [mutual_information(p1X, pX1, p11) for p1X, pX1 in zip(p1Xs, pX1s)]

        # if any(value < 0 for value in result):
        #     print(f"The feature {HotEncoding.feature_from_hot_encoding(feature, ppi.search_space)} has {result}")

        return result

    def faster_linkage_score_for_feature(self, feature: HotEncodedFeature,
                                         ppi: PrecomputedPopulationInformation,
                                         p11: float) -> float:
        present_vals = [val_pos for val_pos, is_used in enumerate(feature) if is_used]

        if len(present_vals) == 0:
            return 0

        if p11 < 1e-6:
            return 0

        def without_that_val(val_pos: int) -> HotEncodedFeature:
            without = np.array(feature)
            without[val_pos] = 0.0
            return without

        f1Xs = list(map(without_that_val, present_vals))

        p1Xs: FlatArray = self.get_proportions_of_features(f1Xs, ppi)
        all_pX1s: FlatArray = self.cached_pX1s.get_data_for_ppi(ppi)
        pX1s = all_pX1s[
            np.array(feature, dtype=bool)]  # feature is used as the predicate to select the marginal probabilities

        max_pX1_p1X = np.max(pX1s * p1Xs)
        return p11 * np.log2(p11 / max_pX1_p1X)

    def all_linkage_scores_for_feature(self, feature: HotEncodedFeature,
                                       ppi: PrecomputedPopulationInformation,
                                       p11: float):
        present_vals = [val_pos for val_pos, is_used in enumerate(feature) if is_used]

        if len(present_vals) == 0:
            return []

        def without_that_val(val_pos: int) -> HotEncodedFeature:
            without = np.array(feature)
            without[val_pos] = 0.0
            return without

        f1Xs = list(map(without_that_val, present_vals))

        p1Xs: FlatArray = self.get_proportions_of_features(f1Xs, ppi)
        all_pX1s: FlatArray = self.cached_pX1s.get_data_for_ppi(ppi)
        pX1s = [pX1 for pX1, is_used in zip(all_pX1s, feature) if is_used]

        mi = [mutual_information(p1X, pX1, p11) for p1X, pX1 in zip(p1Xs, pX1s)]
        ce = [conditional_entropy(p1X, pX1, p11) for p1X, pX1 in zip(p1Xs, pX1s)]
        je = [joint_entropy(p1X, pX1, p11) for p1X, pX1 in zip(p1Xs, pX1s)]
        vi = [variation_of_information(p1X, pX1, p11) for p1X, pX1 in zip(p1Xs, pX1s)]
        v2 = [variation_of_information_2(p1X, pX1, p11) for p1X, pX1 in zip(p1Xs, pX1s)]

        # if any(value < 0 for value in result):
        #     print(f"The feature {HotEncoding.feature_from_hot_encoding(feature, ppi.search_space)} has {result}")

        return mi, ce, je, vi, v2

    def get_weakest_list_for_feature(self, feature: HotEncodedFeature,
                                     ppi: PrecomputedPopulationInformation,
                                     p11: float) -> float:
        if np.sum(feature) < 1:
            return 0

        linkage_scores = self.linkage_scores_for_feature(feature, ppi, p11)
        return np.min(linkage_scores)  # IMPORTANT

    def __repr__(self):
        return f"WeakestLink"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        ppi = pfi.precomputed_population_information
        normalised_fitnesses: FlatArray = self.cached_normalised_fitnesses.get_data_for_ppi(ppi)
        p11s = utils.weighted_sum_of_rows(pfi.feature_presence_matrix, normalised_fitnesses)

        hot_encoded_features = pfi.feature_matrix.T

        scores = np.array([self.faster_linkage_score_for_feature(feature, ppi, p11)
                           for feature, p11 in zip(hot_encoded_features, p11s)])

        return scores

    def describe_score(self, given_score) -> str:
        return f"Weakest_Link = {given_score:.2f}"


class PairwiseRobustness(MeasurableCriterion):

    def __init__(self):
        pass

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        two_errors_fpm = np.array(pfi.feature_presence_error_matrix == 2, dtype=float)
        mean_for_each_feature = get_mean_of_fuzzy_match_matrix(two_errors_fpm, pfi.fitness_array)
        return mean_for_each_feature

    def __repr__(self):
        return "PairwiseRobustness"

    def describe_score(self, given_score) -> str:
        return f"PairwiseRobustness = {given_score}"


class P00Linkage(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "P00Linkage"

    def describe_score(self, given_score) -> str:
        return f"P00Linkage = {given_score}"

    def get_f00s(self, feature: Feature, hot_feature: HotEncodedFeature,
                 search_space: SearchSpace.SearchSpace) -> np.ndarray:
        present_vars = [index for index, is_used in enumerate(feature.variable_mask)
                        if is_used]

        alternate_encoding_feature = np.array(hot_feature)

        def invert_var(input_array: HotEncodedFeature, var: int):
            start, end = search_space.precomputed_offsets[var:var + 2]
            input_array[start:end] = 1 - input_array[start:end]

        for var_index, is_used in enumerate(feature.variable_mask):
            if not is_used:
                invert_var(alternate_encoding_feature, var_index)

        def with_inverted_vars(var_a: int, var_b: int) -> HotEncodedFeature:
            result = np.array(alternate_encoding_feature)
            invert_var(result, var_a)
            invert_var(result, var_b)
            return result

        f00s = [with_inverted_vars(left, right)
                for left, right in utils.cyclic_pairwise(present_vars)]

        return np.array(f00s)

    def get_individual_raw_score(self, feature: Feature, hot_feature: HotEncodedFeature,
                                 ppi: PrecomputedPopulationInformation) -> float:
        if np.sum(hot_feature) < 2:
            return 0
        f00s: np.ndarray = self.get_f00s(feature, hot_feature, ppi.search_space)

        error_count_matrix = ppi.candidate_matrix @ (1 - f00s).T
        no_errors = np.array(error_count_matrix == 1, dtype=float)
        scores = get_mean_of_fuzzy_match_matrix(no_errors, ppi.fitness_array)

        return np.min(scores)

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        ppi = pfi.precomputed_population_information
        hot_features = pfi.feature_matrix.T
        return np.array([self.get_individual_raw_score(feature, hot_feature, ppi)
                         for feature, hot_feature in zip(pfi.features, hot_features)])




class ClassicLinkage(MeasurableCriterion):
    cached_linkage_table: PPICachedData

    Fitness = float
    Variable = int
    Value = int
    # LinkageTable = map[(Variable, Variable), Fitness]

    def get_where_var_val_is_used(self, var: Variable, val: Value, ppi: PrecomputedPopulationInformation) -> FlatArray:
        position_in_hot_encoded = ppi.search_space.position_in_hot_encoded(var, val)
        used_where = np.array(ppi.candidate_matrix[:, position_in_hot_encoded], dtype=bool)
        return used_where

    def get_fitnesses_for_var_val(self, var: Variable, val: Value, ppi: PrecomputedPopulationInformation) -> FlatArray:
        return ppi.fitness_array[self.get_where_var_val_is_used(var, val, ppi)]

    def get_fitnesses_for_var_val_pair(self, var_a: Variable, val_a: Value,
                                       var_b: Variable, val_b: Value,
                                       ppi: PrecomputedPopulationInformation) -> FlatArray:
        where_a = self.get_where_var_val_is_used(var_a, val_a, ppi)
        where_b = self.get_where_var_val_is_used(var_b, val_b, ppi)
        where_both = np.logical_and(where_a, where_b)
        return ppi.fitness_array[where_both]


    def anova_SS_int_between_vars(self, var_a: Variable, var_b: Variable, ppi: PrecomputedPopulationInformation,
                                    total_mean:float, ss_tot:float) -> float:
        def values_for(var) -> Iterable[int]:
            return range(ppi.search_space.cardinalities[var])
        def ss_factor(var) -> float:
            ss_sum_for_levels = 0
            for val in values_for(var):
                fitnesses = self.get_fitnesses_for_var_val(var, val, ppi)
                ss_sum_for_levels += len(fitnesses)*np.square(np.mean(fitnesses)-total_mean)
            return ss_sum_for_levels

        def ss_e(var_a, val_a, var_b, val_b) -> float:
            observations = self.get_fitnesses_for_var_val_pair(var_a, val_a, var_b, val_b, ppi)
            mean_of_observations = np.mean(observations)
            return np.sum(np.square(observations-mean_of_observations))

        ss_within = np.sum([ss_e(var_a, val_a, var_b, val_b)
                            for val_a in values_for(var_a)
                            for val_b in values_for(var_b)])

        ss_int = ss_tot - ss_factor(var_a) - ss_factor(var_b) - ss_within
        return ss_int


    def normalise_table(self, linkage_table: dict):
        present_values = linkage_table.values()
        position_with_values = list(enumerate(present_values))
        position_with_values.sort(key=utils.second)
        scrambled_positions, sorted_values = utils.unzip(position_with_values)
        scrambled_positions_with_rank = list(enumerate(scrambled_positions))
        scrambled_positions_with_rank.sort(key=utils.first)
        resorted_positions, ranks = utils.unzip(scrambled_positions_with_rank)
        max_rank = len(present_values)-1

        normalised_ranks = [rank / max_rank for rank in ranks]

        return {key: rank for (key, rank) in zip(linkage_table.keys(), normalised_ranks)}



    def anova_SS_int_between_all_requested_pairs(self, list_of_pairs: list[(Variable, Variable)],
                                                 ppi: PrecomputedPopulationInformation):
        total_mean: float = np.mean(ppi.fitness_array)
        ss_tot: float = np.sum(np.square(ppi.fitness_array - total_mean))
        linkage_table = {(var_a, var_b): self.anova_SS_int_between_vars(var_a, var_b, ppi, total_mean, ss_tot)
                         for var_a, var_b in list_of_pairs}

        return linkage_table


    def generate_linkage_table(self, ppi: PrecomputedPopulationInformation):
        amount_of_vars = ppi.search_space.dimensions
        pairs_to_calculate = itertools.combinations(range(amount_of_vars), 2)
        result = self.anova_SS_int_between_all_requested_pairs(pairs_to_calculate, ppi)

        result = self.normalise_table(result)

        view_table = np.zeros((amount_of_vars, amount_of_vars))
        for (var_a, var_b) in result:
            value = result[(var_a, var_b)]
            view_table[var_a][var_b] = value
            view_table[var_b][var_a] = value
        return result


    def __init__(self):
        self.cached_linkage_table = PPICachedData(self.generate_linkage_table)


    def get_linkage_values_for_feature(self, feature: Feature, ppi: PrecomputedPopulationInformation):
        used_values = [index for index, is_used in enumerate(feature.variable_mask) if is_used]
        pairs_to_check = itertools.combinations(used_values, 2)
        linkage_table = self.cached_linkage_table.get_data_for_ppi(ppi)
        return [linkage_table[pair] for pair in pairs_to_check]


    def get_single_linkage_score_for_feature(self, feature: Feature, ppi: PrecomputedPopulationInformation):
        linkages = self.get_linkage_values_for_feature(feature, ppi)
        if len(linkages) < 1:
            return 0
        else:
            return np.min(linkages)



    def get_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        ppi = pfi.precomputed_population_information
        return np.array([self.get_single_linkage_score_for_feature(feature, ppi)
                         for feature in pfi.features])


    def describe_score(self, given_score) -> str:
        return f"The feature has classical linkage value of {given_score}"