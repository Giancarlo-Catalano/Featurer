import SearchSpace
import utils
from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem
from Version_E.Feature import Feature


class PlateauProblem(TestableCombinatorialProblem):
    amount_of_groups: int
    k = 4

    def __init__(self, amount_of_groups: int):
        self.amount_of_groups = amount_of_groups
        amount_of_bits = self.k * self.amount_of_groups
        super().__init__(SearchSpace.SearchSpace([2] * amount_of_bits))

    def __repr__(self):
        return f"Plateau({self.k}, {self.amount_of_groups})"

    def long_repr(self):
        return f"Plateau(K={self.k}, amount of groups = {self.amount_of_groups})"

    def get_complexity_of_feature(self, feature: SearchSpace.UserFeature):
        return super().amount_of_set_values_in_feature(feature)

    def score_of_candidate(self, candidate: SearchSpace.Candidate):

        def score_for_group(group_index):
            return int(all(candidate.values[i] for i in range(self.k*group_index, self.k*(group_index+1))))

        return sum(score_for_group(group_index) for group_index in range(self.amount_of_groups))


    def feature_repr(self, feature):
        return f"{Feature.from_legacy_feature(feature, self.search_space)}"

    def get_ideal_features(self) -> list[Feature]:

        def get_ideal_for_group(group_index):
            var_vals = [(var, 1) for var in range(self.k*group_index, self.k*(group_index+1))]
            return Feature.from_legacy_feature(SearchSpace.UserFeature(var_vals), self.search_space)

        return [get_ideal_for_group(group_index) for group_index in range(self.amount_of_groups)]
