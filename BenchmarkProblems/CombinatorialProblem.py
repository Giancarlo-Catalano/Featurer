import SearchSpace
import utils
from Version_E.Feature import Feature


class CombinatorialProblem:
    """ A minimal definition involves

        - repr of the problem
        - long repr of the problem

        - calculating the search space
            (in __init__(...): { search_space = ...; super(search_space)})

        -feature_repr(SearchSpace.Feature) -> str

        -score_of_candidate(SearchSpace.Candidate) -> float

        -get_complexity_of_feature(SearchSpace.Feature) -> float


    """

    search_space: SearchSpace.SearchSpace

    def feature_repr(self, feature) -> str:
        raise Exception("A class extending CombinatorialProblem does not implement .feature_repr(f)!")

    def get_complexity_of_feature(self, feature: SearchSpace.UserFeature):
        """Returns a value which increases as the complexity increases"""
        raise Exception("A class extending CombinatorialProblem does not implement .get_complexity_of_feature(f)!")

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        """Returns a score for the candidate solution"""
        raise Exception("A class extending CombinatorialProblem does not implement .score_of_candidate(c)!")

    def __init__(self, search_space):
        self.search_space = search_space
        pass

    def __repr__(self):
        return f"Generic Combinatorial Problem"

    def long_repr(self):
        return f"Long Description of the Combinatorial Problem"

    def get_random_candidate_solution(self) -> SearchSpace.Candidate:
        return self.search_space.get_random_candidate()

    def get_amount_of_bits_when_hot_encoded(self):
        return self.search_space.total_cardinality

    def amount_of_set_values_in_feature(self, feature: SearchSpace.UserFeature):
        return len(feature.var_vals)

    def get_leftmost_and_rightmost_set_variable(self, feature):
        if len(feature.var_vals) == 0:
            return 0, 0

        used_vars = utils.unzip(feature.var_vals)[0]
        return min(used_vars), max(used_vars)

    def get_area_of_smallest_bounding_box(self, feature):
        leftmost, rightmost = self.get_leftmost_and_rightmost_set_variable(feature)
        return rightmost - leftmost + 1

    def get_positional_values(self, feature: SearchSpace.UserFeature):
        positional_values = [None] * self.search_space.dimensions
        for var, val in feature.var_vals:
            positional_values[var] = val
        return positional_values

    def candidate_repr(self, candidate) -> str:
        return self.feature_repr(candidate.as_feature())


Ideal = Feature
Ideals = list[Ideal]


class TestableCombinatorialProblem(CombinatorialProblem):

    def get_ideal_features(self) -> Ideals:
        raise Exception("An implementation of TestableCombinatorialProblem does not implement get_ideal_features")
