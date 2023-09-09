import itertools

import SearchSpace
import BenchmarkProblems.CombinatorialProblem
import numpy as np
import random


class GraphColouringProblem(BenchmarkProblems.CombinatorialProblem.CombinatorialProblem):
    amount_of_colours: int
    amount_of_nodes: int
    chance_of_connection: float

    adjacency_matrix: np.array

    colours = ["red", "blue", "green", "white", "orange", "purple", "black", "white"]

    def generate_adjacency_matrix(self):
        def should_connect():
            return random.random() > self.chance_of_connection
        result_list = np.array([[should_connect() for _ in range(self.amount_of_nodes)]
                                for _ in range(self.amount_of_nodes)])
        asymmetric = np.array(result_list, dtype=int)
        upper_triangle = np.triu(asymmetric)
        symmetric = upper_triangle+upper_triangle.T - np.diag(upper_triangle)
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
        return self.colours[colour_number]

    def repr_of_node(self, node_number):
        return f"#{node_number}"

    def __repr__(self):
        return f"GraphColouring(nodes={self.amount_of_nodes}, colours = {self.amount_of_colours})"

    def long_repr(self):
        def repr_pair(x, y):
            return f"({x}, {y})"

        return "Connected pairs:" + ", ".join([repr_pair(*pair) for pair in self.connected_pairs])

    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        """returns area of bounding box / area of board"""
        amount_of_set_vars = super().amount_of_set_values_in_feature(feature)
        amount_of_distinct_colours = len(set([val for var, val in feature.var_vals]))
        return (amount_of_distinct_colours/self.amount_of_colours) + (abs(3-amount_of_set_vars)/self.amount_of_nodes)*2

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
