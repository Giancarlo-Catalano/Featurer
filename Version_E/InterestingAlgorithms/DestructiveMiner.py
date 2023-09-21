from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import LayeredFeatureMiner, FeatureSelector
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation


class DestructiveMiner(LayeredFeatureMiner):
    at_least_parameters: int

    def __init__(self, selector: FeatureSelector, amount_to_keep_in_each_layer: int, stochastic: bool,
                 at_least_parameters: int):
        super().__init__(selector, amount_to_keep_in_each_layer, stochastic)
        self.at_least_parameters = at_least_parameters

    def get_initial_features(self, ppi: PrecomputedPopulationInformation) -> list[Feature]:
        return Feature.candidate_matrix_to_features(ppi.candidate_matrix, self.search_space)

    def branch_from_feature(self, feature: Feature) -> list[Feature]:
        return feature.get_generalisations()

    def should_terminate(self, next_iteration: int):
        next_amount_of_parameters = self.search_space.dimensions - next_iteration
        return next_amount_of_parameters < self.at_least_parameters
