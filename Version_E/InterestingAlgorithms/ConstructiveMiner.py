from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import LayeredFeatureMiner, FeatureSelector
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation


class ConstructiveMiner(LayeredFeatureMiner):
    at_most_parameters: int

    def __init__(self, selector: FeatureSelector, amount_to_keep_in_each_layer: int, stochastic: bool,
                 at_most_parameters: int):
        super().__init__(selector, amount_to_keep_in_each_layer, stochastic)
        self.at_most_parameters = at_most_parameters

    def get_initial_features(self, ppi: PrecomputedPopulationInformation) -> list[Feature]:
        return [Feature.empty_feature(self.search_space)]

    def branch_from_feature(self, feature: Feature) -> list[Feature]:
        return feature.get_specialisations(self.search_space)

    def should_terminate(self, next_iteration: int):
        next_amount_of_parameters = next_iteration
        return next_amount_of_parameters > self.at_most_parameters
