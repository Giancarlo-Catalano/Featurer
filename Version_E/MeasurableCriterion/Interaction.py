from typing import Iterable

import numpy as np

import SearchSpace
import utils
from Version_E import HotEncoding
from Version_E.Feature import Feature
from Version_E.MeasurableCriterion.CriterionUtilities import PPICachedData
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
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


def get_pairs_to_check(search_space: SearchSpace.SearchSpace) -> list[((int, int), (int, int))]:
    amount_of_vars = search_space.dimensions

    var_val_quartets = [((var_a, val_a), (var_b, val_b))
                        for var_a in range(amount_of_vars)
                        for val_a in range(search_space.cardinalities[var_a])
                        for var_b in range(var_a + 1, amount_of_vars)
                        for val_b in range(search_space.cardinalities[var_b])]

    return var_val_quartets


def get_normalised_fitnesses(ppi: PrecomputedPopulationInformation) -> FlatArray:
    in_zero_one = utils.remap_array_in_zero_one(ppi.fitness_array)
    with_sum_one = in_zero_one / sum(in_zero_one)
    return with_sum_one


def get_just_value_and_without_just_value(var: int,
                                          val: int,
                                          search_space: SearchSpace.SearchSpace) -> (
        HotEncodedFeature, HotEncodedFeature):
    just_that_value = np.zeros(search_space.total_cardinality, dtype=float)
    just_that_value[search_space.cardinalities[var]+val] = 1.0

    not_that_value = np.array(just_that_value)
    start_pos, end_pos = search_space.precomputed_offsets[var], search_space.precomputed_offsets[var + 1]
    not_that_value[start_pos:end_pos] = 1 - not_that_value[start_pos:end_pos]

    return not_that_value, just_that_value


# FBB is F00, F01, F10, F11...I wanted to call it FXX but that makes it really confusing
FBB = (HotEncodedFeature, HotEncodedFeature, HotEncodedFeature, HotEncodedFeature)

Var_Val = (int, int)


def get_all_f00_f01_f10_f11(pairs_to_generate,  # for some reason I can't put the type information
                            search_space: SearchSpace.SearchSpace) -> list[HotEncodedFeature]:
    """ for each value in the search space, we generate the f00..f11 features,
    which are hot encoded and placed in successive rows.
    Ie, the first four rows are f00..f11 for the first pair of values, and so on"""

    features_with_and_without = [[get_just_value_and_without_just_value(var, val, search_space)
                                  for val in range(search_space.cardinalities[var])]
                                 for var in range(search_space.dimensions)]

    # features_with_and_without[var][val][0 if without, 1 if with]
    def get_f00_f01_f10_f11_for_val_pair(var_a, val_a, var_b, val_b) -> FBB:
        without_a, with_a = features_with_and_without[var_a][val_a]
        without_b, with_b = features_with_and_without[var_b][val_b]

        f00 = without_a + without_b
        f01 = without_a + with_b
        f10 = with_a + without_b
        f11 = with_a + with_b

        return f00, f01, f10, f11

    all_fbbs = []
    for var_val_a, var_val_b in pairs_to_generate:
        var_a, val_a = var_val_a
        var_b, val_b = var_val_b
        fbb = get_f00_f01_f10_f11_for_val_pair(var_a, val_a, var_b, val_b)
        all_fbbs.extend(fbb)

    return all_fbbs


def get_pseudo_proportions_for_hot_encoded_features(hot_encoded_features: Iterable[HotEncodedFeature],
                                                    ppi: PrecomputedPopulationInformation) -> np.ndarray:
    # we'll want to get some stats out of these features, so we create a pfi
    pfi = PrecomputedFeatureInformation.get_from_hot_encoded_features(ppi, np.array(hot_encoded_features))

    # we're trying to find the average normalised fitness, so let's normalise the fitness scores
    pfi.precomputed_population_information.fitness_array = get_normalised_fitnesses(ppi)

    return pfi.mean_fitness_for_each_feature


def mutual_information(p00: float, p01: float, p10: float, p11: float) -> float:
    p0X = p00 + p01
    p1X = p10 + p11
    pX0 = p00 + p10
    pX1 = p01 + p11

    def aux_mutual_information(pAB, pAX, pXB) -> float:
        """https://en.wikipedia.org/wiki/Mutual_information"""
        denominator = pAX * pXB
        if denominator < 1e-6:
            return 1  # panic
        return pAB * np.log2(pAB / denominator)

    return aux_mutual_information(p00, p0X, pX0) + \
        aux_mutual_information(p01, p0X, pX1) + \
        aux_mutual_information(p10, p1X, pX0) + \
        aux_mutual_information(p11, p1X, pX1)


def joint_entropy(p00: float, p01: float, p10: float, p11: float) -> float:
    def entropy(pxy: float) -> float:
        if pxy <= 1e-6:
            return 0.0
        return -pxy * np.log2(pxy)

    return entropy(p00) + entropy(p01) + entropy(p10) + entropy(p11)


def get_interaction_table(ppi: PrecomputedPopulationInformation) -> Matrix:
    print("Starting to generate the table")
    pairs_to_calculate = get_pairs_to_check(ppi.search_space)

    # note that of all the interactions, we only keep the upper triangle, without even the diagonal, and only between different variables
    amount_of_vals = ppi.search_space.total_cardinality

    all_fbb = get_all_f00_f01_f10_f11(pairs_to_calculate, ppi.search_space)
    pseudo_proportions = get_pseudo_proportions_for_hot_encoded_features(all_fbb, ppi)

    pseudo_proportions = pseudo_proportions.reshape((-1, 4))
    table = np.zeros((amount_of_vals, amount_of_vals))

    def flatten_var_val(var_val) -> int:
        var, val = var_val
        return ppi.search_space.precomputed_offsets[var] + val

    for a_b, pbb in zip(pairs_to_calculate, pseudo_proportions):
        var_val_a, var_val_b = a_b
        position_a = flatten_var_val(var_val_a)
        position_b = flatten_var_val(var_val_b)

        p00, p01, p10, p11 = pbb

        value_to_put_in_table = mutual_information(p00, p01, p10, p11)  # might divide it by the joint entropy...
        table[position_a][position_b] = value_to_put_in_table

    print("The table has been generated")
    return table


def get_interactions_for_feature(feature: HotEncodedFeature,
                                 interaction_table: InteractionTable) -> FlatArray:
    boolean_feature = np.array(feature, dtype=bool)
    present_pairs = np.outer(boolean_feature, boolean_feature)
    present_pairs = np.triu(present_pairs)

    return np.select(present_pairs, interaction_table)


def get_interaction_score_for_feature(feature: HotEncodedFeature,
                                      interaction_table: InteractionTable) -> InteractionScore:
    return np.max(get_interactions_for_feature(feature, interaction_table))  # perhaps the minimum would be better?


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
    return np.squared(pfi.sd_for_each_feature)


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

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        variances = np.square(pfi.sd_for_each_feature)

        divorce_scores = np.array([get_divorce_score(feature, variance, pfi)
                                   for feature, variance in zip(pfi.features, variances)])

        return divorce_scores

    def describe_score(self, given_score) -> str:
        return f"Divorce = {given_score}"
