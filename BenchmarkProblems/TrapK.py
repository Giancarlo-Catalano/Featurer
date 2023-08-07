import SearchSpace
import utils
import BenchmarkProblems.CombinatorialProblem


class TrapK(BenchmarkProblems.CombinatorialProblem.CombinatorialProblem):
    k: int
    amount_of_groups: int
    amount_of_bits: int

    def __init__(self, k, amount_of_groups):
        self.k = k
        self.amount_of_groups = amount_of_groups
        self.amount_of_bits = k * amount_of_groups
        super().__init__(SearchSpace.SearchSpace([2] * self.amount_of_bits))


    def __repr__(self):
        return f"TrapK(K={self.k}, amount of groups = {self.amount_of_groups})"

    def get_how_many_vars_per_group(self, feature: SearchSpace.Feature):
        how_many_per_group = [0]* self.amount_of_groups
        for var, _ in feature.var_vals:
            how_many_per_group[var // self.amount_of_groups] += 1
        return how_many_per_group

    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        return utils.product([area + 1 for area in self.get_how_many_vars_per_group(feature)])

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
        for group in self.divide_candidate_in_groups(SearchSpace.Candidate(super().get_positional_values(feature))):
            print("[", end="")
            for cell in group:
                print(f"{cell_repr(cell)} ", end="")
            print("]")



