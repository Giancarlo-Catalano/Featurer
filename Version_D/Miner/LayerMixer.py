import itertools

import utils
from SearchSpace import SearchSpace
from Version_D.Feature import Feature
from Version_D.MeasurableCriterion import MeasurableCriterion, compute_scores_for_features, LayerScoringCriteria
from Version_D.Miner.MinerLayer import MinerLayer

import numpy as np
import random

from Version_D.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_D.PrecomputedPopulationInformation import PrecomputedPopulationInformation


class ParentPairIterator:
    mother_layer: MinerLayer
    father_layer: MinerLayer

    ppi: PrecomputedPopulationInformation

    def __repr__(self):
        raise Exception("An implementation of ParentPairIterator does not implement __repr__")

    def to_code(self):
        raise Exception("An implementation of ParentPairIterator does not implement to_code")

    def setup(self):
        raise Exception("An implementation of ParentPairIterator does not implement setup")

    def reset(self) -> bool:
        raise Exception("An implementation of ParentPairIterator does not implement reset")

    def set_parents(self, mother_layer: MinerLayer, father_layer: MinerLayer):
        # this acts as a delayed init function
        self.mother_layer = mother_layer
        self.father_layer = father_layer

    def set_ppi(self, ppi: PrecomputedPopulationInformation):
        self.ppi = ppi

    def get_next_parent_pair(self) -> (Feature, Feature):
        raise Exception("An implementation of ParentPairIterator does not implement get_next_parent_pair")

    def is_finished(self) -> bool:
        raise Exception("An implementation of ParentPairIterator does not implement is_finished")


def get_layer_offspring(mother_layer: MinerLayer,
                        father_layer: MinerLayer,
                        ppi: PrecomputedPopulationInformation,
                        parent_pair_iterator: ParentPairIterator,
                        requested_amount: int) -> list[Feature]:
    avoid_overlap = True

    if len(mother_layer.features) == 0 or len(father_layer.features) == 0:
        raise Exception("Attempting to construct a layer mixer from empty parents!")

    parent_pair_iterator.set_parents(mother_layer, father_layer)
    parent_pair_iterator.set_ppi(ppi)
    parent_pair_iterator.setup()

    accumulator = list()

    def add_child_if_allowed(parents: (Feature, Feature)):
        mother, father = parents
        if avoid_overlap and Feature.overlap(mother, father):
            return
        accumulator.append(Feature.merge(mother, father))

    def fill_accumulator():
        while len(accumulator) < requested_amount:
            if parent_pair_iterator.is_finished():
                break
            add_child_if_allowed(parent_pair_iterator.get_next_parent_pair())

    fill_accumulator()
    # if len(accumulator) < requested_amount:
    #    avoid_overlap = False
    #    fill_accumulator()

    return list(set(accumulator))


class TotalSearchIterator(ParentPairIterator):
    """ Will systematically search produce all the offspring"""
    mother_index: int
    father_index: int

    amount_of_mothers: int
    amount_of_fathers: int

    identical_parents: bool

    def to_code(self):
        return "T"

    def __repr__(self):
        return "Total Search"

    def __init__(self):
        pass

    def increase_indices_when_parents_are_different(self):
        self.mother_index += 1
        if self.mother_index >= self.amount_of_mothers:
            self.mother_index = 0
            self.father_index += 1

    def increase_indices_when_parents_are_same(self):
        self.mother_index += 1
        if self.mother_index >= self.amount_of_mothers:
            self.father_index += 1
            self.mother_index = self.father_index + 1

    def increase_indices(self):
        if self.identical_parents:
            self.increase_indices_when_parents_are_same()
        else:
            self.increase_indices_when_parents_are_different()

        # print(f"The next indices are {self.mother_index}, {self.father_index}")

    def get_next_parent_pair(self) -> (Feature, Feature):
        # print(f"Accessing indexes {self.mother_index}, {self.father_index}")
        result = (self.mother_layer.features[self.mother_index],
                  self.father_layer.features[self.father_index])
        self.increase_indices()
        return result

    def is_finished(self) -> bool:
        return (self.father_index >= self.amount_of_fathers) or (self.mother_index >= self.amount_of_mothers)

    def reset(self):
        # if the parents are identical, start from 0,1, else start from 0, 0
        self.mother_index = 1 if self.identical_parents else 0
        self.father_index = 0

    def setup(self):
        self.amount_of_mothers = len(self.mother_layer.features)
        self.amount_of_fathers = len(self.father_layer.features)
        self.identical_parents = (self.mother_layer is self.father_layer)
        self.reset()


class GreedyHeuristicIterator(ParentPairIterator):
    pairs_to_visit: list[(int, int)]
    visit_index = 0

    def to_code(self):
        return "H"

    def __repr__(self):
        return "Greedy Heuristic"

    def compute_pairs_to_visit(self, mother_layer: MinerLayer, father_layer: MinerLayer) -> list[(int, int)]:

        if mother_layer is father_layer:
            all_pairs = itertools.combinations(range(mother_layer.amount_of_features), r=2)
        else:
            all_pairs = itertools.product(range(mother_layer.amount_of_features),
                                          range(father_layer.amount_of_features))

        all_pairs_with_scores = [((mother_index, father_index),
                                  mother_layer.scores[mother_index] + father_layer.scores[father_index])
                                 for mother_index, father_index in all_pairs]
        all_pairs_with_scores.sort(key=utils.second, reverse=True)  # sort by which pair has the highest sum of scores
        return utils.unzip(all_pairs_with_scores)[0]

    def __init__(self):
        pass

    def get_next_parent_pair(self) -> (Feature, Feature):
        mother_index, father_index = self.pairs_to_visit[self.visit_index]
        self.visit_index += 1
        return (self.mother_layer.features[mother_index],
                self.father_layer.features[father_index])

    def is_finished(self) -> bool:
        return self.visit_index >= len(self.pairs_to_visit)

    def reset(self):
        self.visit_index = 0

    def setup(self):
        self.pairs_to_visit = self.compute_pairs_to_visit(self.mother_layer, self.father_layer)


class StochasticIterator(ParentPairIterator):
    mother_cumulative_weights: np.array
    father_cumulative_weights: np.array

    batch_size = 256
    current_batch: list[(Feature, Feature)]
    index_within_batch: int

    def to_code(self):
        "S"

    def __repr__(self):
        return "Stochastic"

    def generate_batch(self) -> list[(Feature, Feature)]:
        mothers = random.choices(self.mother_layer.features, self.mother_cumulative_weights, k=self.batch_size)
        fathers = random.choices(self.father_layer.features, self.father_cumulative_weights, k=self.batch_size)
        return list(zip(mothers, fathers))

    def __init__(self):
        pass

    def setup(self):
        self.mother_cumulative_weights = np.cumsum(self.mother_layer.scores)
        self.father_cumulative_weights = np.cumsum(self.father_layer.scores)
        self.current_batch = self.generate_batch()
        self.reset()

    def reset(self):
        self.index_within_batch = 0

    def is_finished(self) -> bool:
        return False

    def get_next_parent_pair(self) -> (Feature, Feature):
        result = self.current_batch[self.index_within_batch]
        self.index_within_batch += 1
        if self.index_within_batch >= self.batch_size:
            self.reset()

        return result


class MatrixIterator(ParentPairIterator):
    pairs_to_visit: list[(int, int)]
    visit_index: int

    def to_code(self):
        return "M"

    def __repr__(self):
        return "Matrix Iterator"

    def __init__(self):
        pass

    def get_correlation_matrix(self):
        mother_pfi = PrecomputedFeatureInformation(self.ppi, self.mother_layer.features)
        father_pfi = PrecomputedFeatureInformation(self.ppi, self.father_layer.features)

        def scale_each_row(matrix, scalings):
            return (matrix.T * scalings).T

        mother_matrix = mother_pfi.feature_presence_matrix
        father_matrix = father_pfi.feature_presence_matrix

        scaled_mother = scale_each_row(mother_matrix, mother_pfi.fitness_array)
        scaled_father = scale_each_row(father_matrix, father_pfi.fitness_array)

        numerator = (scaled_mother.T @ scaled_father)
        denominator = (mother_matrix.T @ father_matrix)
        return utils.divide_arrays_safely(numerator, denominator, else_value=0)

    def get_order_of_cells(self, matrix: np.ndarray):
        amount_of_rows, amount_of_cols = matrix.shape
        cell_coords = np.array([(row, col) for row in range(amount_of_rows) for col in range(amount_of_cols)])
        parents_are_identical = (self.mother_layer is self.father_layer)

        cell_coord_and_score = None
        if parents_are_identical:
            cell_coord_and_score = [(coords, score) for coords, score in zip(cell_coords, np.nditer(matrix))
                                    if coords[1] > coords[0]]
        else:
            cell_coord_and_score = [(coords, score) for coords, score in zip(cell_coords, np.nditer(matrix))]

        cell_coord_and_score.sort(key=utils.second, reverse=True)
        cell_order = utils.unzip(cell_coord_and_score)[0]
        return cell_order

    def setup(self):
        correlation_matrix = self.get_correlation_matrix()
        self.pairs_to_visit = self.get_order_of_cells(correlation_matrix)
        self.reset()

    def reset(self):
        self.visit_index = 0

    def get_next_parent_pair(self) -> (Feature, Feature):
        next_mom_index, next_dad_index = self.pairs_to_visit[self.visit_index]
        self.visit_index += 1
        return self.mother_layer.features[next_mom_index], self.father_layer.features[next_dad_index]

    def is_finished(self) -> bool:
        return self.visit_index >= len(self.pairs_to_visit)


def make_0_parameter_layer(search_space: SearchSpace) -> MinerLayer:
    empty_feature = Feature.empty_feature(search_space)
    scores = np.array(1)  # dummy value
    return MinerLayer([empty_feature], ppi=None, scores=scores)


def make_1_parameter_layer(ppi: PrecomputedPopulationInformation,
                           criteria_and_weights: LayerScoringCriteria) -> MinerLayer:
    # create
    features = Feature.get_all_trivial_features(ppi.search_space)

    # assess
    pfi: PrecomputedFeatureInformation = PrecomputedFeatureInformation(ppi, features)
    scores = compute_scores_for_features(pfi, criteria_and_weights)

    # don't select!
    return MinerLayer(features, ppi, np.array(scores))


def make_layer_by_mixing(mother_layer, father_layer,
                         ppi: PrecomputedPopulationInformation,
                         criteria_and_weights: LayerScoringCriteria,
                         parent_pair_iterator: ParentPairIterator,
                         how_many_to_generate: int,
                         how_many_to_keep: int) -> MinerLayer:
    # breed
    offspring = get_layer_offspring(mother_layer, father_layer, ppi,
                                    parent_pair_iterator, requested_amount=how_many_to_generate)
    # assess
    pfi: PrecomputedFeatureInformation = PrecomputedFeatureInformation(ppi, offspring)
    scores = compute_scores_for_features(pfi, criteria_and_weights)

    # select
    sorted_by_with_score = sorted(zip(offspring, scores), key=utils.second, reverse=True)
    features, scores_list = utils.unzip(sorted_by_with_score[:how_many_to_keep])
    return MinerLayer(features, ppi, np.array(scores_list))
