import utils
from Version_D import MeasurableCriterion, Feature
from Version_D.Miner.MinerLayer import MinerLayer
from Version_D.Miner import LayerMixer, Parameters
from Version_D.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_D.PrecomputedPopulationInformation import PrecomputedPopulationInformation
import numpy as np


def mine_meaningful_features(ppi: PrecomputedPopulationInformation,
                             parameter_schedule: Parameters.Schedule) -> (list[Feature], np.ndarray):
    layers = []
    layers.append(LayerMixer.make_0_parameter_layer(ppi.search_space))
    layers.append(LayerMixer.make_1_parameter_layer(ppi, parameter_schedule[1].criteria_and_weights))

    def get_parent_layers(child_weight: int):
        mother_weight = child_weight // 2 # if child_weight % 2 == 0 else child_weight-1
        father_weight = child_weight - mother_weight

        return layers[mother_weight], layers[father_weight]

    def get_next_layer(weight, parameters: Parameters.IterationParameters):
        mother, father = get_parent_layers(weight)
        new_layer = LayerMixer.make_layer_by_mixing(mother, father, ppi,
                                                    criteria_and_weights=parameters.criteria_and_weights,
                                                    parent_pair_iterator=parameters.mixing_iterator,
                                                    how_many_to_generate=parameters.generated_feature_amount,
                                                    how_many_to_keep=parameters.kept_feature_amount)

        layers.append(new_layer)

    for weight, parameters in enumerate(parameter_schedule[2:], start=2):
        print(f"For weight = {weight} and parameters = {parameters}, we generate the layer.")
        get_next_layer(weight, parameters)

    all_features = utils.concat_lists([layer.features for layer in layers])
    final_pfi = PrecomputedFeatureInformation(ppi, all_features)
    final_criteria = parameter_schedule[-1].criteria_and_weights
    scores = MeasurableCriterion.compute_scores_for_features(final_pfi, final_criteria)

    features_and_scores = list(zip(all_features, scores))
    features_and_scores.sort(key=utils.second, reverse=True)
    all_features, scores = utils.unzip(features_and_scores)  # here you would truncate

    return all_features, scores
