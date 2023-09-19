import random

import SearchSpace
import BenchmarkProblems.CombinatorialProblem
from typing import List


class ArtificialProblem(BenchmarkProblems.CombinatorialProblem.CombinatorialProblem):
    """In this problem, the fitness function is given by the presence of some features"""
    """ To avoid issues, these features are guaranteed to be non-overlapping, meaning that all features are disjoint"""
    """ In other terms, no particular var-val combination is present in more than one feature. """
    """ BUT NOTE: the same var might be present in multiple features, just with different values"""

    amount_of_bits: int

    amount_of_partials: int
    size_of_partials: int

    important_partials: map[SearchSpace.Feature, int]
    def generate_non_overlapping_partials(self):

        def get_random_feature():
            value = random.randrange(2)
            which_vars = random.sample(range(self.amount_of_bits), k=self.size_of_partials)
            feature = SearchSpace.Feature([(var, value) for var in which_vars])
            return feature

        def are_disjoint(feature_a: SearchSpace.Feature, feature_b: SearchSpace.Feature):
            var_vals_a = set(feature_a.var_vals)
            var_vals_b = set(feature_b.var_vals)
            overlap = var_vals_a.intersection(var_vals_b)
            return len(overlap) == 0

        accumulator = []
        def is_eligible(new_feature: SearchSpace.Feature):
            return all([are_disjoint(old_feature, new_feature) for old_feature in accumulator])

        while len(accumulator) <= self.amount_of_partials:
            new_feature = get_random_feature()
            if is_eligible(new_feature):
                accumulator.append(new_feature)

        return accumulator

    def __init__(self, amount_of_bits, cardinality, amount_of_partials, size_of_partials):
        self.amount_of_bits = amount_of_bits
        self.amount_of_partials = amount_of_partials
        self.size_of_partials = size_of_partials
        super().__init__(SearchSpace.SearchSpace([2] * self.amount_of_bits))
        self.important_partials = self.generate_non_overlapping_partials()

    def __repr__(self):
        return (f"ArtificialProblem(bits={self.amount_of_bits}, "
                f"amount_of_partials = {self.amount_of_partials},"
                f"size_of_partials = {self.size_of_partials}")



    def long_repr(self):
        return self.__repr__()

    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        return super().get_area_of_smallest_bounding_box(feature)

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        pass

    def feature_repr(self, feature):
        def cell_repr(cell):
            return "-" if cell is None else str(cell)

        return " ".join([cell_repr(cell) for cell in super().get_positional_values(feature)])
