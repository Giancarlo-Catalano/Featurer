import random

import numpy as np
from bitarray import frozenbitarray
from bitarray.util import urandom

from SearchSpace import SearchSpace
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureSelector, FeatureMiner


def random_feature_in_search_space(search_space: SearchSpace) -> Feature:
    used_cells = frozenbitarray(urandom(search_space.dimensions))
    values = np.zeros(search_space.dimensions, dtype=int)
    for variable, is_used in enumerate(used_cells):
        if is_used:
            values[variable] = random.randrange(search_space.cardinalities[variable])

    return Feature(used_cells, values)


class RandomSearch(FeatureMiner):
    amount_to_generate: int

    def __init__(self, selector: FeatureSelector, amount_to_generate):
        super().__init__(selector)
        self.amount_to_generate = amount_to_generate

    def mine_features(self) -> list[Feature]:
        return [random_feature_in_search_space(self.search_space) for _ in range(self.amount_to_generate)]
