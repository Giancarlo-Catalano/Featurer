import math

import SearchSpace
import utils
import numpy as np


class TrapK:
    k: int
    amount_of_groups: int

    amount_of_bits: int

    def __init__(self, k, amount_of_groups):
        self.k = k
        self.amount_of_groups = amount_of_groups
        self.amount_of_bits = k*amount_of_groups

    def __repr__(self):
        return f"TrapK(K={self.k}, amount of groups = {self.amount_of_groups})"

    def get_search_space(self):
        return SearchSpace.SearchSpace([2] * self.amount_of_bits)

    def get_bounding_box(self, feature):
        used_columns = [col for (col, feature_cell) in enumerate(feature.values)
                             if feature_cell is not None]

        if len(used_columns) == 0:
            return (0, 0)
        return (min(used_columns), max(used_columns)+1)


    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        """returns area of bounding box / area of board"""
        bounds = self.get_bounding_box(feature)
        bounds_area = bounds[1]-bounds[0]
        normal_score = bounds_area/self.amount_of_bits
        return 0.20*(normal_score-0.5)+0.5


    def divide_candidate_in_groups(self, candidate: SearchSpace.Candidate):
        return [candidate.values[which*self.k:(which+1)*self.k] for which in range(self.amount_of_groups)]

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        def score_of_group(group):
            amount_of_bits = sum(group)
            if (amount_of_bits == self.k):
                return self.k
            else:
                return (self.k-1)-amount_of_bits

        groups = self.divide_candidate_in_groups(candidate)
        return sum([score_of_group(g) for g in groups])

    def pretty_print_feature(self, feature):
        def cell_repr(cell):
            if cell is None:
                return "_"
            else:
                return str(cell)
        for group in self.divide_candidate_in_groups(feature):
            print("[", end="")
            for cell in group:
                print(f"{cell_repr(cell)} ", end="")
            print("]")



