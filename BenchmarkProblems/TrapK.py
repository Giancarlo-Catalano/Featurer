import SearchSpace
import utils
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem
from Version_E.Feature import Feature


class TrapK(TestableCombinatorialProblem):
    k: int
    amount_of_groups: int
    amount_of_bits: int

    def __init__(self, k, amount_of_groups):
        self.k = k
        self.amount_of_groups = amount_of_groups
        self.amount_of_bits = k * amount_of_groups
        super().__init__(SearchSpace.SearchSpace([2] * self.amount_of_bits))

    def __repr__(self):
        return f"TrapK({self.k}, {self.amount_of_groups})"

    def long_repr(self):
        return f"TrapK(K={self.k}, amount of groups = {self.amount_of_groups})"

    def get_how_many_vars_per_group(self, feature: SearchSpace.UserFeature):
        how_many_per_group = [0] * self.amount_of_groups
        for var, _ in feature.var_vals:
            how_many_per_group[var // self.k] += 1
        return how_many_per_group

    def get_complexity_of_feature(self, feature: SearchSpace.UserFeature):
        return super().amount_of_set_values_in_feature(feature)

    def divide_candidate_in_groups(self, candidate: SearchSpace.Candidate):
        return [candidate.values[which * self.k:(which + 1) * self.k] for which in range(self.amount_of_groups)]

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        def score_of_group(group):
            amount_of_ones = sum(group)
            if amount_of_ones == self.k:
                return self.k
            else:
                return (self.k - 1) - amount_of_ones

        groups = self.divide_candidate_in_groups(candidate)
        return sum([score_of_group(g) for g in groups])

    def feature_repr(self, feature):
        def cell_repr(cell):
            if cell is None:
                return "_"
            else:
                return str(cell)

        def group_repr(group):
            return "[" + (" ".join([cell_repr(cell) for cell in group])) + "]"

        groups = self.divide_candidate_in_groups(SearchSpace.Candidate(super().get_positional_values(feature)))
        return "\n".join([group_repr(group) for group in groups])

    def get_all_ones_ideals(self) -> list[Feature]:
        def all_ones_in_group(group_index) -> Feature:
            start = group_index * self.k
            end = start + self.k
            return Feature.from_legacy_feature(SearchSpace.UserFeature([(var, 1) for var in range(start, end)]),
                                               search_space=self.search_space)

        return [all_ones_in_group(group_index) for group_index in range(self.amount_of_groups)]

    def get_all_zero_ideals(self) -> list[Feature]:
        empty_feature = Feature.empty_feature(self.search_space)

        def feature_with_a_single_zero(var_index) -> Feature:
            return empty_feature.with_value(var_index, 0)

        return [feature_with_a_single_zero(var_index) for var_index in range(self.amount_of_bits)]

    def get_ideal_features(self) -> list[Feature]:
        deceptive_groups = self.get_all_ones_ideals()
        zeros = self.get_all_zero_ideals()
        return deceptive_groups  # + zeros
