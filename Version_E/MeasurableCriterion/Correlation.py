import numpy as np

import utils
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation


def compute_phi_scores(pfi: PrecomputedFeatureInformation):
    count_for_each_value = np.sum(pfi.candidate_matrix, axis=0)
    absence_count_for_each_value = pfi.sample_size - count_for_each_value

    counts_everywhere = np.tile(count_for_each_value, (pfi.amount_of_features, 1))
    absences_everywhere = np.tile(absence_count_for_each_value, (pfi.amount_of_features, 1))
    value_is_used_in_feature = np.array(pfi.feature_matrix.T, dtype=bool)

    products_of_counts = np.product(counts_everywhere, where=value_is_used_in_feature, axis=1)

    products_of_absences = np.product(absences_everywhere, where=value_is_used_in_feature, axis=1)

    n = pfi.sample_size
    n_all = pfi.count_for_each_feature

    numerators = (n * n_all - products_of_counts)
    denominators = np.sqrt(products_of_counts * products_of_absences)

    return utils.divide_arrays_safely(numerators, denominators, 0)


class CorrelationCriterion(MeasurableCriterion):
    def __init__(self):
        pass

    def __repr__(self):
        return "Correlation"

    def get_raw_score_array(self, pfi: PrecomputedFeatureInformation) -> np.ndarray:
        return compute_phi_scores(pfi)


    def describe_score(self, given_score) -> str:
        return f"Correlation with phi-score = {given_score}"

