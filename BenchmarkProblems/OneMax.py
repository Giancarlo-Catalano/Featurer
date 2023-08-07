import SearchSpace
import BenchmarkProblems.CombinatorialProblem


class OneMaxProblem(BenchmarkProblems.CombinatorialProblem.CombinatorialProblem):
    amount_of_bits: int

    def __init__(self, amount_of_bits):
        self.amount_of_bits = amount_of_bits
        super().__init__(SearchSpace.SearchSpace([2] * self.amount_of_bits))


    def __repr__(self):
        return f"OneMaxProblem(bits={self.amount_of_bits})"

    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        """returns area of bounding box / area of board"""
        return super().amount_of_set_values_in_feature(feature)

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        return sum(candidate.values)

    def pretty_print_feature(self, feature):
        def cell_repr(cell):
            if cell is None:
                return "_"
            else:
                return str(cell)

        for cell in super().get_positional_values(feature):
            print(f"{cell_repr(cell)} ", end="")
