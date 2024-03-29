import SearchSpace
import utils
from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from Version_E.Feature import Feature


class CheckerBoardProblem(CombinatorialProblem):
    rows: int
    cols: int

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        super().__init__(SearchSpace.SearchSpace([2] * self.total_area))

    def __repr__(self):
        return f"CheckerBoard(rows={self.rows}, cols={self.cols})"

    def long_repr(self):
        return self.__repr__()

    @property
    def total_area(self):
        return self.rows * self.cols

    def get_bounding_box(self, feature):
        def var_index_to_cell_coords(var):
            return divmod(var, self.cols)

        used_cells = [var_index_to_cell_coords(var) for var, _ in feature.var_vals]

        (rows_used, cols_used) = utils.unzip(used_cells)
        return min(rows_used), max(rows_used) + 1, \
            min(cols_used), max(cols_used) + 1

    def get_complexity_of_feature(self, feature: SearchSpace.UserFeature):
        """returns area of bounding box / area of board"""

        return super().amount_of_set_values_in_feature(feature)

    def candidate_to_boolean_grid(self, candidate: SearchSpace.Candidate):
        candidate_bools = [val == 1 for val in candidate.values]
        return [candidate_bools[row * self.cols:(row + 1) * self.cols] for row in range(self.rows)]

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        def are_opposite(a, b):
            return int(a != b)

        candidate_as_grid = self.candidate_to_boolean_grid(candidate)

        total_score = 0
        for row in range(self.rows):
            for col in range(self.cols):
                current_cell = candidate_as_grid[row][col]
                if col != self.cols - 1:
                    cell_right = candidate_as_grid[row][col + 1]
                    total_score += are_opposite(current_cell, cell_right)
                if row != self.rows - 1:
                    cell_down = candidate_as_grid[row + 1][col]
                    total_score += are_opposite(current_cell, cell_down)

        return total_score

    def feature_repr(self, feature):
        positional_values = super().get_positional_values(feature)
        as_grid = [positional_values[row * self.cols:(row + 1) * self.cols] for row in range(self.rows)]

        def cell_repr(cell):
            if cell is None:
                return "_"
            else:
                return str(cell)

        def row_repr(row):
            return " ".join([cell_repr(cell) for cell in row])

        return "\n".join([row_repr(row) for row in as_grid])


    def get_ideal_feature(self, input_row, input_col, ul_most_value:int, horizontal: bool):
        def coords_to_index(row, col):
            return row*self.rows + col


        main_cell = (coords_to_index(input_row, input_col), ul_most_value)
        other_cell_position = coords_to_index(input_row, input_col+1) if horizontal else coords_to_index(input_row+1, input_col)
        other_cell = (other_cell_position, 1-ul_most_value)

        return Feature.from_legacy_feature(SearchSpace.UserFeature([main_cell, other_cell]), self.search_space)

    def get_ideal_features(self) -> list[Feature]:
        horizontals = [self.get_ideal_feature(row, col, value, horizontal=True) for row in range(self.rows)
                       for col in range(self.cols-1)
                       for value in range(2)]
        verticals = [self.get_ideal_feature(row, col, value, horizontal=False) for row in range(self.rows-1)
                     for col in range(self.cols)
                     for value in range(2)]
        return horizontals + verticals