import SearchSpace
import CombinatorialProblem
import utils


class BinValProblem(CombinatorialProblem.CombinatorialProblem):
    amount_of_bits: int
    base: float

    def __init__(self, amount_of_bits, base):
        super().__init__(SearchSpace.SearchSpace([2] * self.amount_of_bits))
        self.amount_of_bits = amount_of_bits
        self.base = base

    def __repr__(self):
        return f"BinValProblem(bits={self.amount_of_bits}, base = {self.base})"

    def get_bounding_box(self, feature):
        if len(feature.var_vals) == 0:
            return (0, 0)
        used_columns = utils.unzip(feature.var_vals)
        return (min(used_columns), max(used_columns)+1)

    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        """returns area of bounding box / area of board"""
        if len(feature.var_vals) == 0:
            return 0
        bounds = self.get_bounding_box(feature)
        bounds_area = bounds[1]-bounds[0]
        return bounds_area

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

