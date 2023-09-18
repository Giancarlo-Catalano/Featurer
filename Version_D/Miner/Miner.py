import utils
from MinerLayer import MinerLayer
from Version_D import MeasurableCriterion
from Version_D import Feature
import Parameters
from Version_D.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_D.PrecomputedPopulationInformation import PrecomputedPopulationInformation
import numpy as np


def mine_meaningful_features(ppi: PrecomputedPopulationInformation,
                             parameter_schedule: Parameters.Schedule) -> (list[Feature], np.ndarray):
    layers = []
    layers.append(MinerLayer.make_0_parameter_layer(ppi.search_space))
    layers.append(MinerLayer.make_1_parameter_layer(ppi,
                                                    parameter_schedule[1].criteria_and_weights,
                                                    parameter_schedule[1].mixing_iterator))

    def get_parent_layers(weight: int):
        mother_weight = weight // 2
        father_weight = weight - mother_weight

        return layers[mother_weight], layers[father_weight]

    def get_next_layer(weight, parameters: Parameters.IterationParameters):
        mother, father = get_parent_layers(weight)
        new_layer = MinerLayer.make_by_mixing(mother, father, ppi,
                                              criteria_and_weights=parameters.criteria_and_weights,
                                              parent_pair_iterator=parameters.mixing_iterator,
                                              how_many_to_generate=parameters.generated_feature_amount,
                                              how_many_to_keep=parameters.kept_feature_amount)

        layers.append(new_layer)

    for weight, parameters in enumerate(parameter_schedule, start=2):
        get_next_layer(weight, parameters)

    all_features = utils.concat_lists([layer.features for layer in layers])
    final_pfi = PrecomputedFeatureInformation(ppi, all_features)
    final_criteria = parameter_schedule[-1].criteria_and_weights
    scores = MeasurableCriterion.compute_scores_for_features(final_pfi, final_criteria)

    return all_features, scores
