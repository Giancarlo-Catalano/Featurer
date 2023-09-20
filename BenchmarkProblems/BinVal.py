import SearchSpace
import BenchmarkProblems.CombinatorialProblem
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem


class BinValProblem(TestableCombinatorialProblem):
    amount_of_bits: int
    base: float

    def __init__(self, amount_of_bits, base):
        self.amount_of_bits = amount_of_bits
        self.base = base
        super().__init__(SearchSpace.SearchSpace([2] * self.amount_of_bits))

    def __repr__(self):
        return f"BinValProblem(bits={self.amount_of_bits}, base = {self.base})"

    def long_repr(self):
        return self.__repr__()

    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        return super().get_area_of_smallest_bounding_box(feature)

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        digit_value = 1
        total = 0
        for cell_value in reversed(candidate.values):
            total += digit_value * cell_value
            digit_value *= self.base
        return total

    def feature_repr(self, feature):
        def cell_repr(cell):
            return "-" if cell is None else str(cell)

        return " ".join([cell_repr(cell) for cell in super().get_positional_values(feature)])

    def get_ideal_features(self) -> list[SearchSpace.Feature]:
        def feature_with_a_single_one(var_index):
            return SearchSpace.Feature([(var_index, 1)])

        return [feature_with_a_single_one(var_index) for var_index in range(self.amount_of_bits)]
