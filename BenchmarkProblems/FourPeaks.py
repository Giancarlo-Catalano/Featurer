import SearchSpace
import BenchmarkProblems.CombinatorialProblem
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem


class FourPeaksProblem(TestableCombinatorialProblem):
    amount_of_bits: int
    t: int

    def __init__(self, amount_of_bits, t: int):
        self.amount_of_bits = amount_of_bits
        self.t = t
        super().__init__(SearchSpace.SearchSpace([2] * self.amount_of_bits))

    def __repr__(self):
        return f"BinValProblem(bits={self.amount_of_bits}, t = {self.t})"

    def long_repr(self):
        return self.__repr__()

    def get_complexity_of_feature(self, feature: SearchSpace.UserFeature):
        return super().get_area_of_smallest_bounding_box(feature)

    def head(self, candidate: SearchSpace.Candidate, value: int) -> int:
        for index, int_value in enumerate(candidate.values):
            if int_value != value:
                return index
        return len(candidate.values)

    def tail(self, candidate: SearchSpace.Candidate, value: int) -> int:
        reversed_candidate = SearchSpace.Candidate(reversed(candidate.values))
        return self.head(reversed_candidate, 0)
    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        """taken directly from https://apps.dtic.mil/sti/pdfs/ADA322735.pdf"""
        head_1 = self.head(candidate, 1)
        tail_0 = self.tail(candidate, 0)

        reward = 100 if (head_1 > self.t) and (tail_0 > self.t) else 0
        return max(head_1, tail_0) + reward

    def feature_repr(self, feature):
        def cell_repr(cell):
            return "-" if cell is None else str(cell)

        return "".join([cell_repr(cell) for cell in super().get_positional_values(feature)])

    def get_ideal_features(self) -> list[SearchSpace.UserFeature]:
        """TODO"""
        def feature_with_a_single_one(var_index):
            return SearchSpace.UserFeature([(var_index, 1)])

        return [feature_with_a_single_one(var_index) for var_index in range(self.amount_of_bits)]
