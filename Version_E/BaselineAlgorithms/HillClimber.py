import copy

from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, FeatureSelector
from RandomSearch import random_feature_in_search_space


class HillClimber(FeatureMiner):
    amount_to_generate: int

    def __init__(self, selector: FeatureSelector, amount_to_generate):
        super().__init__(selector)
        self.amount_to_generate = amount_to_generate

    def get_mutations_of_feature(self, feature: Feature) -> list[Feature]:
        return (feature.get_generalisations() +
                feature.get_specialisations(self.search_space) +
                feature.get_variations(self.search_space))

    def get_best_mutation_of_feature(self, feature: Feature) -> (Feature, float):
        mutations = self.get_mutations_of_feature(feature)
        return self.feature_selector.keep_best_features(mutations, 1)[0]

    def improve_feature_until_stationary(self, feature: Feature) -> Feature:
        current_best_feature = copy.copy(feature)
        current_best_score = self.feature_selector.get_scores([current_best_feature])[0]

        while True:
            new_feature, new_score = self.get_best_mutation_of_feature(current_best_feature)
            if new_score > current_best_score:
                current_best_feature = new_feature
                current_best_score = new_score
            else:
                break

        return current_best_feature

    def mine_features(self) -> list[Feature]:
        def mine_feature():
            start = random_feature_in_search_space(self.search_space)
            return self.improve_feature_until_stationary(start)

        return [mine_feature() for _ in range(self.amount_to_generate)]
