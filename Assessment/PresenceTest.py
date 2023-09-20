from BenchmarkProblems.CombinatorialProblem import TestableCombinatorialProblem, which_ideals_are_present
from SearchSpace import Feature

PresenceDict = dict[Feature, bool]


def count_how_many_were_present(problem: TestableCombinatorialProblem, presence_dict: PresenceDict):
    how_many_in_total = len(presence_dict)
    how_many_present = sum(presence_dict.values())
    return (how_many_present, how_many_in_total)



