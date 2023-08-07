import SearchSpace


class CombinatorialProblem:


    search_space: SearchSpace.SearchSpace
    def __init__(self, search_space):
        self.search_space = search_space
        pass

    def __repr__(self):
        return f"Generic Combinatorial Problem"

    def get_amount_of_bits(self):
        return self.search_space.total_cardinality

    def amount_of_set_values_in_feature(self, feature: SearchSpace.Feature):
        return len(feature.var_vals)

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

