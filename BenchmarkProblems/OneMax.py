import SearchSpace
import BenchmarkProblems.CombinatorialProblem
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem


class OneMaxProblem(TestableCombinatorialProblem):
    amount_of_bits: int

    def __init__(self, amount_of_bits):
        self.amount_of_bits = amount_of_bits
        super().__init__(SearchSpace.SearchSpace([2] * self.amount_of_bits))

    def __repr__(self):
        return f"OneMaxProblem(bits={self.amount_of_bits})"


    def long_repr(self):
        return self.__repr__()

    def get_complexity_of_feature(self, feature: SearchSpace.UserFeature):
        """returns area of bounding box / area of board"""
        return super().amount_of_set_values_in_feature(feature)

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        return sum(candidate.values)

    def feature_repr(self, feature):
        def cell_repr(cell):
            if cell is None:
                return "_"
            else:
                return str(cell)

        return " ".join([cell_repr(cell) for cell in super().get_positional_values(feature)])

    def get_ideal_features(self) -> list[SearchSpace.UserFeature]:
        def feature_with_a_single_one(var_index):
            return SearchSpace.UserFeature([(var_index, 1)])

        return [feature_with_a_single_one(var_index) for var_index in range(self.amount_of_bits)]
