import numpy as np

import SearchSpace
import utils
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


def which_candidates_have_value(val_pos: ValuePosition, ppi: PrecomputedPopulationInformation) -> ColumnVector:
    return np.array(ppi.candidate_matrix[:, val_pos], dtype=bool)


def get_list_of_fitnesses_for_val(val_pos: ValuePosition, ppi: PrecomputedPopulationInformation) -> list[Fitness]:
    which = which_candidates_have_value(val_pos, ppi)
    return ppi.fitness_array[which]


def get_list_of_fitnesses_for_pair_of_vals(val_pos_a: ValuePosition, val_pos_b: ValuePosition,
                                           ppi: PrecomputedPopulationInformation) -> list[Fitness]:
    has_a = which_candidates_have_value(val_pos_a, ppi)
    has_b = which_candidates_have_value(val_pos_b, ppi)

    return ppi.fitness_array[has_a & has_b]


def get_interaction_score(val_pos_a: ValuePosition,
                          val_pos_b: ValuePosition,
                          ppi: PrecomputedPopulationInformation,
                          sds_for_each_val: FlatArray) -> InteractionScore:
    sd_a = sds_for_each_val[val_pos_a]
    sd_b = sds_for_each_val[val_pos_b]
    fitnesses_for_ab = get_list_of_fitnesses_for_pair_of_vals(val_pos_a, val_pos_b, ppi)
    sd_ab = np.std(fitnesses_for_ab)

    return sd_ab / (sd_a * sd_b)


def get_pairs_to_check(search_space: SearchSpace.SearchSpace) -> list[(int, int)]:
    def var_val_to_index(var, val):
        return search_space.precomputed_offsets[var] + val

    amount_of_vars = search_space.dimensions

    return [(var_val_to_index(var_a, val_a),
             var_val_to_index(var_b, val_b))
            for var_a in range(amount_of_vars)
            for val_a in range(search_space.cardinalities[var_a])
            for var_b in range(var_a + 1, amount_of_vars)
            for val_b in range(search_space.cardinalities[var_b])]


def get_interaction_table(ppi: PrecomputedPopulationInformation) -> Matrix:
    pairs_to_calculate = get_pairs_to_check(ppi.search_space)

    # note that of all the interactions, we only keep the upper triangle, without even the diagonal, and only between different variables
    amount_of_vals = ppi.search_space.total_cardinality
    sds_for_each_val = [np.std(get_list_of_fitnesses_for_val(val_pos, ppi))
                        for val_pos in range(amount_of_vals)]
    result = np.zeros((amount_of_vals, amount_of_vals))
    for val_a, val_b in pairs_to_calculate:
        result[val_a][val_b] = get_interaction_score(val_a, val_b, ppi, sds_for_each_val)

    for val_pos in range(amount_of_vals):
        result[val_pos][val_pos] = 1 / sds_for_each_val[val_pos]

    return result


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
                            for without, only, count_of_without in zip(features_without_var, just_vars, temp_pfi.count_for_each_feature)
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
