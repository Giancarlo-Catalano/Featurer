import numpy as np

import utils
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation


FlatArray = np.ndarray
Matrix = np.ndarray
ColumnVector = np.ndarray

ValuePosition = int   #the index
Fitness = float

InteractionScore = float
InteractionTable = Matrix

HotEncodedFeature = FlatArray

def which_candidates_have_value(val_pos: ValuePosition, ppi: PrecomputedPopulationInformation) -> ColumnVector:
    return np.array(ppi.candidate_matrix[:][val_pos], dtype=bool)
def get_list_of_fitnesses_for_val(val_pos: ValuePosition, ppi: PrecomputedPopulationInformation) -> list[Fitness]:
    return ppi.fitness_array[which_candidates_have_value(val_pos, ppi)]


def get_list_of_fitnesses_for_pair_of_vals(val_pos_a: ValuePosition, val_pos_b: ValuePosition,
                                           ppi: PrecomputedPopulationInformation) -> list[Fitness]:
    has_a = which_candidates_have_value(val_pos_a, ppi)
    has_b = which_candidates_have_value(val_pos_b, ppi)

    return ppi.fitness_array[has_a & has_b]


def get_interaction_score(val_pos_a: ValuePosition,
                          val_pos_b: ValuePosition,
                          ppi: PrecomputedPopulationInformation) -> InteractionScore:
    sd_a = np.std(get_list_of_fitnesses_for_val(val_pos_a, ppi))
    sd_b = np.std(get_list_of_fitnesses_for_val(val_pos_b, ppi))
    sd_ab = np.std(get_list_of_fitnesses_for_pair_of_vals(val_pos_a, val_pos_b, ppi))

    return sd_ab / (sd_a * sd_b)


def get_interaction_table(ppi: PrecomputedPopulationInformation) -> Matrix:
    amount_of_vars = ppi.search_space.total_cardinality
    pairs_to_calculate = [(a, b)
                          for a in range(amount_of_vars)
                          for b in range(a+1, amount_of_vars)]

    # note that of all the interactions, we only keep the upper triangle, without even the diagonal

    result = np.zeros((amount_of_vars, amount_of_vars))
    for val_a, val_b in pairs_to_calculate:
        result[val_a][val_b] = get_interaction_score(val_a, val_b, ppi)

    return result


def get_interactions_for_feature(feature: HotEncodedFeature,
                                 interaction_table: InteractionTable) -> FlatArray:
    boolean_feature = np.array(feature, dtype=bool)
    present_pairs = np.outer(boolean_feature, boolean_feature)
    present_pairs = np.triu(present_pairs)
    present_pairs ^= present_pairs.diagonal()

    return np.select(present_pairs, interaction_table)


def get_interaction_score_for_feature(feature: HotEncodedFeature,
                                      interaction_table: InteractionTable) -> FlatArray:
    return np.average(get_interactions_for_feature(feature, interaction_table))  # perhaps the minimum would be better?




# TODO
#   cache an interaction table for each ppi
#   when raw scores are needed, calculate using that cached interaction table
#   Implement a proper MeasurableCriterion



