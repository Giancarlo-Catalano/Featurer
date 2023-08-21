import SearchSpace
import utils


class CombinatorialProblem:


    search_space: SearchSpace.SearchSpace
    def __init__(self, search_space):
        self.search_space = search_space
        pass

    def __repr__(self):
        return f"Generic Combinatorial Problem"


    def get_random_candidate_solution(self):
        return self.search_space.get_random_candidate()

    def get_amount_of_bits_when_hot_encoded(self):
        return self.search_space.total_cardinality

    def amount_of_set_values_in_feature(self, feature: SearchSpace.Feature):
        return len(feature.var_vals)


    def get_leftmost_and_rightmost_set_variable(self, feature):
        if len(feature.var_vals) == 0:
            return 0, 0

        used_vars = utils.unzip(feature.var_vals)[0]
        return min(used_vars), max(used_vars)

    def get_area_of_smallest_bounding_box(self, feature):
        leftmost, rightmost = self.get_leftmost_and_rightmost_set_variable(feature)
        return rightmost-leftmost+1

    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        """Returns a value which increases as the complexity increases"""
        # It should never be negative!
        raise Exception("A class extending CombinatorialProblem does not implement .get_complexity_of_feature(f)!")

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        """Returns a score for the candidate solution"""
        raise Exception("A class extending CombinatorialProblem does not implement .score_of_candidate(c)!")


    def get_positional_values(self, feature: SearchSpace.Feature):
        positional_values = [None]*self.search_space.dimensions
        for var, val in feature.var_vals:
            positional_values[var] = val
        return positional_values

    def pretty_print_feature(self, feature):
        raise Exception("A class extending CombinatorialProblem does not implement .pretty_print_feature(f)!")


    def pretty_print_candidate(self, candidate):
        self.pretty_print_feature(candidate.as_feature())