import copy
import random

from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureMiner, FeatureSelector
from RandomSearch import random_feature_in_search_space


class GAMiner(FeatureMiner):
    population_size: int
    iterations: int

    tournament_size = 30
    chance_of_mutation = 0.05


    def __init__(self, selector: FeatureSelector, population_size: int, iterations: int):
        super().__init__(selector)
        self.population_size = population_size
        self.iterations = iterations


    def mutate(self, feature: Feature) -> Feature:
        result = copy.copy(feature)
        pass
