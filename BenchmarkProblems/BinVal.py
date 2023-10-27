import SearchSpace
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem
from Version_E.Feature import Feature


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

    def get_complexity_of_feature(self, feature: SearchSpace.UserFeature):
        return self.amount_of_set_values_in_feature(feature)

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        digit_value = 1
        total = 0
        for cell_value in reversed(candidate.values):
            total += digit_value * cell_value
            digit_value *= self.base
        return total

    def feature_repr(self, feature):
        return f"{Feature.from_legacy_feature(feature, self.search_space)}"

    def get_ideal_features(self) -> list[Feature]:
        empty_feature = Feature.empty_feature(self.search_space)

        def feature_with_a_single_one(var_index):
            return empty_feature.with_value(var_index, 1)

        return [feature_with_a_single_one(var_index) for var_index in range(self.amount_of_bits)]
