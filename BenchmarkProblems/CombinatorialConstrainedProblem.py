from BenchmarkProblems import CombinatorialProblem
import SearchSpace


class CombinatorialConstrainedProblem(CombinatorialProblem.CombinatorialProblem):
    """This class is a combinatorial problem which also has some predicates included in the candidate solutions"""
    """This allows the feature exploration to also find relationships which include these predicates"""

    """Note how this class is 
      - derived from CombinatorialProblem, because it still has the same functionalities
      - contains an instance of CombinatorialProblem, because it's like a stove lighter containing a bic lighter"""

    """Minimal implementation:
        - the unconstrained problem, to be passed in the constructor
        - satisfying some requirements of a combinatorial problem
            - repr
            - long_repr
            - score_of_candidate
            - get_complexity_of_feature
        - obtaining the constraint space
            in __init__(...): 
                unconstrained_problem = ...
                constraint_space = ...
                super().__init(unconstrained_problem, constraint_space)
        - get_predicates(SearchSpace.Candidate) -> SearchSpace.Candidate
        - predicate_feature_repr(SearchSpace.Candidate) -> str
        
    
    """

    unconstrained_problem: CombinatorialProblem.CombinatorialProblem
    search_space: SearchSpace.SearchSpace

    def __repr__(self):
        raise Exception("A class extending CombinatorialConstraintProblem does not implement __repr__(self)!")


    def long_repr(self):
        raise Exception("A class extending CombinatorialConstraintProblem does not implement long_repr")

    def get_predicates(self, candidate_solution: SearchSpace.Candidate) -> SearchSpace.Candidate:
        raise Exception("A class extending CombinatorialConstraintProblem does not implement .get_predicates(c)!")

    def predicate_feature_repr(self, predicates: SearchSpace.Feature) -> str:
        raise Exception("A class extending CombinatorialConstraintProblem "
                        "does not implement .predicate_feature_repr_(c)!")

    @property
    def dimensions_in_unconstrained_space(self) -> int:
        return self.unconstrained_problem.search_space.dimensions

    def split_feature(self, feature: SearchSpace.Feature) -> (SearchSpace.Feature, SearchSpace.Feature):
        parameter_var_vals = []
        predicates_var_vals = []

        for var, val in feature.var_vals:
            if var < self.dimensions_in_unconstrained_space:
                parameter_var_vals.append((var, val))
            else:
                predicates_var_vals.append((var-self.dimensions_in_unconstrained_space, val))

        parameters = SearchSpace.Feature(parameter_var_vals)
        predicates = SearchSpace.Feature(predicates_var_vals)

        return parameters, predicates

    def split_candidate(self, candidate: SearchSpace.Candidate) -> (SearchSpace.Candidate, SearchSpace.Candidate):
        without_predicates = SearchSpace.Candidate(candidate.values[:self.dimensions_in_unconstrained_space])
        predicates = SearchSpace.Candidate(candidate.values[self.dimensions_in_unconstrained_space:])
        return without_predicates, predicates

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
        # , and then we add the predicates
        solution_without_predicates = self.unconstrained_problem.get_random_candidate_solution()
        extension = self.get_predicates(solution_without_predicates)
        return SearchSpace.merge_two_candidates(solution_without_predicates, extension)

    def feature_repr(self, feature: SearchSpace.Feature) -> str:
        parameters, predicates = self.split_feature(feature)
        return (f"{self.unconstrained_problem.feature_repr(parameters)}\n"
                f" {self.predicate_feature_repr(predicates)}")

    def score_of_candidate(self, candidate: SearchSpace.Candidate) -> float:
        raise Exception("A class extending CombinatorialConstraintProblem does not implement .score_of_candidate(c)!")


    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        return super().amount_of_set_values_in_feature(feature)
