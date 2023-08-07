import SearchSpace
import BenchmarkProblems.CombinatorialProblem


class BinValProblem(BenchmarkProblems.CombinatorialProblem.CombinatorialProblem):
    amount_of_bits: int
    base: float

    def __init__(self, amount_of_bits, base):
        self.amount_of_bits = amount_of_bits
        self.base = base
        super().__init__(SearchSpace.SearchSpace([2] * self.amount_of_bits))


    def __repr__(self):
        return f"BinValProblem(bits={self.amount_of_bits}, base = {self.base})"

    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        return super().get_area_of_smallest_bounding_box(feature)

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        digit_value = 1
        total = 0
        for cell_value in reversed(candidate.values):
            total += digit_value*cell_value
            digit_value*=self.base
        return total

    def pretty_print_feature(self, feature):
        def cell_repr(cell):
            return "-" if cell is None else str(cell)

        for cell in super().get_positional_values(feature):
            print(f"{cell_repr(cell)} ", end="")

