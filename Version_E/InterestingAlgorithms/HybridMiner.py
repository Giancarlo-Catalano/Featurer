import math

import utils
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import LayeredFeatureMiner, FeatureSelector, FeatureMiner
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation


class HybridMiner(FeatureMiner):
    population_size: int
    stochastic: bool

    def __init__(self, selector: FeatureSelector, population_size: int, stochastic: bool):
        super().__init__(selector)
        self.population_size = population_size
        self.stochastic = stochastic

    def __repr__(self):
        return (f"Hybrid(population = {self.population_size}, "
                f"stochastic = {self.stochastic}")

    def select_parents(self, current_population: list[(Feature, float)]) -> list[Feature]:
        portion_to_select = 0.5
        amount_to_select = math.ceil(len(current_population) * portion_to_select)
        if self.stochastic:
            return self.feature_selector.select_features_stochastically(current_population, amount_to_select)
        else:
            return self.feature_selector.select_features_heuristically(current_population, amount_to_select)

    def get_next_population(self, prev_population: list[(Feature, float)]) -> list[(Feature, float)]:
        selected_parents = self.select_parents(prev_population)
        simplifications = {simplified for feature in selected_parents
                           for simplified in feature.get_generalisations()}

        complications = {complected for feature in selected_parents
                         for complected in feature.get_specialisations(self.search_space)}

        prev_population_as_set = {feature for feature, score in prev_population}
        unselected_new_population = simplifications.union(complications).union(prev_population_as_set)
        return self.feature_selector.keep_best_features(unselected_new_population, self.population_size)


    def get_initial_population(self) -> list[(Feature, float)]:
        initial_features = [Feature.empty_feature(self.search_space)]
        scores = self.feature_selector.get_scores(initial_features)
        return list(zip(initial_features, scores))

    def get_max_score_of_population(self, population: list[(Feature, float)]) -> float:
        return max(population, key = utils.second)[1]

    def mine_features(self) -> list[Feature]:
        current_population = self.get_initial_population()
        current_best = self.get_max_score_of_population(current_population)


        allowed_successive_fails = 5
        current_successive_fails = 0

        while current_successive_fails < allowed_successive_fails:
            current_population = self.get_next_population(current_population)
            new_best = self.get_max_score_of_population(current_population)
            if new_best <= current_best:
                current_successive_fails +=1
            else:
                current_successive_fails = 0

            current_best = new_best

        return current_population


