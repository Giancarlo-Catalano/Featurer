import utils
from Version_D.Feature import Feature
from Version_D.Miner.MinerLayer import MinerLayer

import numpy as np
import random


class ParentPairIterator:
    mother_layer: MinerLayer
    father_layer: MinerLayer

    def reset(self) -> bool:
        raise Exception("An implementation of ParentPairIterator does not implement reset")

    def set_parents(self, mother_layer: MinerLayer, father_layer: MinerLayer):
        # this acts as a delayed init function
        self.mother_layer = mother_layer
        self.father_layer = father_layer
        self.reset()

    def get_next_parent_pair(self) -> (Feature, Feature):
        raise Exception("An implementation of ParentPairIterator does not implement get_next_parent_pair")

    def is_finished(self) -> bool:
        raise Exception("An implementation of ParentPairIterator does not implement is_finished")


def get_layer_offspring(mother_layer: MinerLayer,
                        father_layer: MinerLayer,
                        parent_pair_iterator: ParentPairIterator,
                        requested_amount: int) -> list[Feature]:
    avoid_overlap = True

    if len(mother_layer.features) == 0 or len(father_layer.features) == 0:
        raise Exception("Attempting to construct a layer mixer from empty parents!")

    parent_pair_iterator.set_parents(mother_layer, father_layer)

    accumulator = set()

    def add_child_if_allowed(parents: (Feature, Feature)):
        mother, father = parents
        if avoid_overlap and Feature.overlap(mother, father):
            return
        accumulator.add(Feature.merge(mother, father))

    def fill_accumulator():
        while len(accumulator) < requested_amount:
            if parent_pair_iterator.is_finished():
                break
            add_child_if_allowed(parent_pair_iterator.get_next_parent_pair())

    fill_accumulator()
    if len(accumulator) < requested_amount:
        avoid_overlap = False
        fill_accumulator()

    return list(accumulator)


class TotalSearchIterator(ParentPairIterator):
    """ Will systematically search produce all of the offspring"""
    mother_index: int
    father_index: int

    amount_of_mothers: int
    amount_of_fathers: int

    def __init__(self):
        self.mother_index = 0  # it increases after iterator access
        self.father_index = 0

    def increase_indices(self):
        self.mother_index += 1
        if self.mother_index >= self.amount_of_mothers:
            self.mother_index = 0
            self.father_index += 1

    def get_next_parent_pair(self) -> (Feature, Feature):
        result = (self.mother_layer.features[self.mother_index],
                  self.father_layer.features[self.father_index])
        self.increase_indices()
        return result

    def is_finished(self) -> bool:
        return self.father_index >= self.amount_of_fathers

    def reset(self):
        self.mother_index = 0
        self.father_index = 0


class GreedyHeuristicIterator(ParentPairIterator):
    pairs_to_visit: list[(int, int)]
    visit_index = 0

    def compute_pairs_to_visit(self, mother_layer: MinerLayer, father_layer: MinerLayer) -> list[(int, int)]:
        pairs_with_scores = [((mother_index, father_index), mother_score + father_score)
                             for ((mother_index, mother_score), (father_index, father_score))
                             in zip(enumerate(mother_layer.features), enumerate(father_layer.features))]
        pairs_with_scores.sort(key=utils.second, reverse=True)  # sort by which pair has the highest sum of scores
        return utils.unzip(pairs_with_scores)[0]

    def __init__(self, mother_layer: MinerLayer, father_layer: MinerLayer):
        super().__init__(mother_layer, father_layer)
        self.pairs_to_visit = self.compute_pairs_to_visit(mother_layer, father_layer)

    def get_next_parent_pair(self) -> (Feature, Feature):
        mother_index, father_index = self.pairs_to_visit[self.visit_index]
        self.visit_index += 1
        return (self.mother_layer.features[mother_index],
                self.father_layer.features[father_index])

    def is_finished(self) -> bool:
        return self.visit_index >= len(self.pairs_to_visit)

    def reset(self):
        self.visit_index = 0


class StochasticIterator(ParentPairIterator):
    mother_cumulative_weights: np.array
    father_cumulative_weights: np.array

    batch_size = 256
    current_batch: list[(Feature, Feature)]
    index_within_batch: int

    def generate_batch(self) -> list[(Feature, Feature)]:
        mothers = random.choices(self.mother_layer.features, self.mother_cumulative_weights, k=self.batch_size)
        fathers = random.choices(self.father_layer.features, self.father_cumulative_weights, k=self.batch_size)
        return list(zip(mothers, fathers))

    def __init__(self, mother_layer: MinerLayer, father_layer: MinerLayer):
        super().__init__(mother_layer, father_layer)
        self.mother_cumulative_weights = np.cumsum(mother_layer.scores)
        self.father_cumulative_weights = np.cumsum(father_layer.scores)

        self.current_batch = self.generate_batch()
        self.index_within_batch = 0

    def reset(self):
        self.current_batch = self.generate_batch()
        self.index_within_batch = 0

    def is_finished(self) -> bool:
        return False

    def get_next_parent_pair(self) -> (Feature, Feature):
        result = self.current_batch[self.index_within_batch]
        self.index_within_batch += 1
        if self.index_within_batch >= self.batch_size:
            self.reset()

        return result
