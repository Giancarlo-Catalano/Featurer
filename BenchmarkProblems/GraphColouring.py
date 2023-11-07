import itertools
from typing import Iterable

import SearchSpace
import BenchmarkProblems.CombinatorialProblem
import numpy as np
import random

from Version_E.Feature import Feature

colours = ["red", "blue", "green", "white", "orange", "purple", "black", "white"]


class GraphColouringProblem(BenchmarkProblems.CombinatorialProblem.CombinatorialProblem):
    amount_of_colours: int
    amount_of_nodes: int
    chance_of_connection: float

    adjacency_matrix: np.array

    def generate_adjacency_matrix(self) -> np.ndarray:
        def should_connect():
            return random.random() > self.chance_of_connection

        result_list = np.array([[should_connect() for _ in range(self.amount_of_nodes)]
                                for _ in range(self.amount_of_nodes)])
        asymmetric = np.array(result_list, dtype=int)
        upper_triangle = np.triu(asymmetric)
        upper_triangle -= np.diag(np.diag(upper_triangle))
        symmetric = upper_triangle + upper_triangle.T
        return symmetric

    def get_connected_pairs_from_adjacency_matrix(self, adjacency_matrix: np.ndarray):
        def are_connected(pair):
            return adjacency_matrix[pair]

        all_pairs = itertools.combinations(range(self.amount_of_nodes), 2)
        connected_pairs = filter(are_connected, all_pairs)
        return list(connected_pairs)

    def __init__(self, amount_of_colours, amount_of_nodes, chance_of_connection):
        self.amount_of_colours = amount_of_colours
        self.amount_of_nodes = amount_of_nodes
        self.chance_of_connection = chance_of_connection
        super().__init__(SearchSpace.SearchSpace([self.amount_of_colours] * self.amount_of_nodes))
        self.adjacency_matrix = self.generate_adjacency_matrix()
        self.connected_pairs = self.get_connected_pairs_from_adjacency_matrix(self.adjacency_matrix)

    def repr_of_colour(self, colour_number):
        return colours[colour_number]

    def repr_of_node(self, node_number):
        return f"#{node_number}"

    def __repr__(self):
        return f"GraphColouring(nodes={self.amount_of_nodes}, colours = {self.amount_of_colours})"

    def long_repr(self):
        def repr_pair(x, y):
            return f"({x}, {y})"

        return "Connected pairs:" + ", ".join([repr_pair(*pair) for pair in self.connected_pairs])

    def get_complexity_of_feature(self, feature: SearchSpace.UserFeature):
        """returns area of bounding box / area of board"""
        amount_of_set_vars = super().amount_of_set_values_in_feature(feature)
        amount_of_distinct_colours = len(set([val for var, val in feature.var_vals]))
        return (amount_of_set_vars / self.amount_of_nodes)  # + (amount_of_distinct_colours/self.amount_of_colours)

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        def are_different_colours(node_x, node_y):
            return candidate.values[node_x] != candidate.values[node_y]

        score = 0
        for (x, y) in self.connected_pairs:
            if are_different_colours(x, y):
                score += 1

        return score

    def feature_repr(self, feature):
        return "\n".join([f"{self.repr_of_node(var)} is {self.repr_of_colour(val)}"
                          for var, val in feature.var_vals])


class InsularGraphColouringProblem(BenchmarkProblems.CombinatorialProblem.CombinatorialProblem):
    Node = int
    Islet = set[Node]

    islets: list[Islet]
    amount_of_colours = 4

    def get_islet_generator(self):
        starting_node = 0
        while True:
            amount_of_nodes = random.choice([3, 4])
            new_starting_node = starting_node + amount_of_nodes
            new_islet = {node for node in range(starting_node, new_starting_node)}
            starting_node = new_starting_node
            yield new_islet

    def generate_islets(self, amount_of_islets) -> list[Islet]:
        islet_generator = self.get_islet_generator()
        return [next(islet_generator) for _ in range(amount_of_islets)]

    def get_search_space_of_islet(self, islet: Islet) -> SearchSpace.SearchSpace:
        return SearchSpace.SearchSpace([self.amount_of_colours for node in islet])

    def __init__(self, amount_of_islets: int):
        self.islets = self.generate_islets(amount_of_islets)
        super().__init__(SearchSpace.merge_many_spaces(self.get_search_space_of_islet(islet) for islet in self.islets))

    def repr_of_colour(self, colour_number):
        return colours[colour_number]

    def repr_of_node(self, node_number):
        return f"#{node_number}"

    def __repr__(self):
        return f"InsularGraphColouring(islets={len(self.islets)}, colours = {self.amount_of_colours})"

    def long_repr(self):
        return "islets:\n\t" + "\n\t".join(f"{islet}" for islet in self.islets)

    def get_complexity_of_feature(self, feature: SearchSpace.UserFeature):
        return super().amount_of_set_values_in_feature(feature)

    def score_for_islet_in_candidate(self, islet: Islet, candidate: SearchSpace.Candidate) -> int:
        def score_for_pair(pair: (int, int)):
            return int(candidate.values[pair[0]] != candidate.values[pair[1]])

        return sum(score_for_pair(pair) for pair in itertools.combinations(islet, 2))

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        return sum(self.score_for_islet_in_candidate(islet, candidate) for islet in self.islets)

    def feature_repr(self, feature: SearchSpace.UserFeature):
        return "\n".join([f"{self.repr_of_node(var)} is {self.repr_of_colour(val)}"
                          for var, val in feature.var_vals])

    def get_islet_of_node(self, node: Node) -> Islet:
        for islet in self.islets:
            if node in islet:
                return islet

        raise Exception(f"Islet could not be found for node {node}")

    def is_ideal_feature(self, feature: Feature) -> bool:
        """ a feature is ideal if it's all the nodes of a single islet"""


        def is_all_the_nodes_of_a_single_islet():
            present_nodes = [node for node, is_used in enumerate(feature.variable_mask) if is_used]

            # handle the edge case of no variables being set
            if len(present_nodes) == 0:
                return False

            islet = self.get_islet_of_node(present_nodes[0])

            if len(islet) != len(present_nodes):
                return False  # needs to contain all of the nodes in the islet

            return all(node in islet for node in present_nodes)

        def all_nodes_have_different_colours():
            present_colours = [val for var, val in feature.to_var_val_pairs()]
            culled_present_colours = set(present_colours)
            return len(culled_present_colours) == len(present_colours)

        return is_all_the_nodes_of_a_single_islet() and all_nodes_have_different_colours()
