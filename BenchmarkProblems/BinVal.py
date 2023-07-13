import math

import SearchSpace
import utils
import numpy as np


class BinValProblem:
    amount_of_bits: int
    base: float

    def __init__(self, amount_of_bits, base):
        self.amount_of_bits = amount_of_bits
        self.base = base

    def __repr__(self):
        return f"BinValProblem(bits={self.amount_of_bits}, base = {self.base})"

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
        return normal_score

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        digit_value = 1
        total = 0
        for cell_value in reversed(candidate.values):
            total += digit_value*cell_value
            digit_value*=self.base
        return total

    def pretty_print_feature(self, feature):
        def cell_repr(cell):
            if cell is None:
                return "_"
            else:
                return str(cell)
        for cell in feature.values:
            print(f"{cell_repr(cell)} ", end="")

