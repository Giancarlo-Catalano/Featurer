import CombinatorialProblem
import SearchSpace


class CombinatorialConstrainedProblem(CombinatorialProblem.CombinatorialProblem):
    """This class is a combinatorial problem which also has some predicates included in the candidate solutions"""
    """This allows the feature exploration to also find relationships which include these predicates"""

    """Note how this class is 
      - derived from CombinatorialProblem, because it still has the same functionalities
      - contains an instance of CombinatorialProblem, because it's like a stove lighter containing a bic lighter"""

    """Minimal implementation:
        - repr
        - the unconstrained problem definition
        - obtaining the constraint space
            in __init__(...): 
                unconstrained_problem = ...
                constraint_space = ...
                super(unconstrained_problem, constraint_space)
        - get_predicates(SearchSpace.Candidate) -> SearchSpace.Candidate
        - constraint_repr(SearchSpace.Candidate) -> str
        
    
    """

    unconstrained_problem: CombinatorialProblem.CombinatorialProblem
    search_space: SearchSpace.SearchSpace

    def __repr__(self):
        pass

    def get_predicates(self, candidate_solution: SearchSpace.Candidate) -> SearchSpace.Candidate:
        pass

    def predicate_repr(self, predicates: SearchSpace.Candidate) -> None:
        pass

    def split_feature(self, feature: SearchSpace.Feature) -> (SearchSpace.Feature, SearchSpace.Feature):
        parameter_var_vals = []
        predicates_var_vals = []
        dimensions_in_unconstrained_space = self.unconstrained_problem.search_space.dimensions

        for var, val in feature.var_vals:
            if var < dimensions_in_unconstrained_space:
                parameter_var_vals.append((var, val))
            else:
                predicates_var_vals.append((var, val))

        parameters = SearchSpace.Feature(parameter_var_vals)
        predicates = SearchSpace.Feature(predicates_var_vals)

        return parameters, predicates

    # from here we implement CombinatorialProblem
    def __init__(self, unconstrained_problem: CombinatorialProblem.CombinatorialProblem,
                 constraint_space: SearchSpace.SearchSpace):

        # the search space is the merging of the unconstrained space and the constraint space
        search_space_with_predicates = SearchSpace.merge_two_spaces(unconstrained_problem.search_space,
                                                                    constraint_space)
        super().__init__(search_space_with_predicates)
        self.unconstrained_problem = unconstrained_problem

    def get_random_candidate_solution(self) -> SearchSpace.Candidate:
        # to get a random solution, we get a random point from the unconstrained problem
        # and then we add the predicates
        solution_without_predicates = self.unconstrained_problem.get_random_candidate_solution()
        extension = self.get_predicates(solution_without_predicates)
        return SearchSpace.merge_two_candidates(solution_without_predicates, extension)

    def feature_repr(self, feature: SearchSpace.Feature) -> str:
        parameters, predicates = self.split_feature(feature)
        return (f"{self.unconstrained_problem.feature_repr(parameters)}\n"
                f" Predicates: {self.predicate_repr(predicates)}")
