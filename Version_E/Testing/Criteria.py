from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from Version_E.MeasurableCriterion.CriterionUtilities import All, Any, Not, Balance, Extreme
from Version_E.MeasurableCriterion.Explainability import Explainability, TrivialExplainability
from Version_E.MeasurableCriterion.ForSampling import Completeness
from Version_E.MeasurableCriterion.GoodFitness import HighFitness, ConsistentFitness, FitnessHigherThanAverage
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.MeasurableCriterion.Popularity import Overrepresentation, Commonality
from Version_E.MeasurableCriterion.Robustness import Robustness


def decode_criterion(properties: dict, problem: CombinatorialProblem) -> MeasurableCriterion:
    criterion_string = properties["which"]

    if criterion_string == "explainability":
        return TrivialExplainability()
    elif criterion_string == "high_fitness":
        return HighFitness()
    elif criterion_string == "low_fitness":
        return Not(HighFitness())
    elif criterion_string == "consistent_fitness":
        return ConsistentFitness(signed=True)
    elif criterion_string == "overrepresentation":
        return Overrepresentation()
    elif criterion_string == "underrepresentation":
        return Not(Overrepresentation())
    elif criterion_string == "fitness_higher_than_average":
        return FitnessHigherThanAverage()
    elif criterion_string == "commonality":
        return Commonality()
    elif criterion_string == "robustness":
        return Robustness(properties["min_diff"], "max_diff")
    elif criterion_string == "not":
        return Not(decode_criterion(properties["argument"], problem))
    elif criterion_string == "extreme":
        return Extreme(decode_criterion(properties["argument"], problem))
    elif criterion_string == "all":
        parsed_arguments = [decode_criterion(criterion, problem) for criterion in properties["arguments"]]
        return All(parsed_arguments)
    elif criterion_string == "any":
        parsed_arguments = [decode_criterion(criterion, problem) for criterion in properties["arguments"]]
        return Any(parsed_arguments)
    elif criterion_string == "balance":
        parsed_arguments = [decode_criterion(criterion, problem) for criterion in properties["arguments"]]
        weights = properties["weights"]
        return Balance(parsed_arguments, weights=weights)
    elif criterion_string == "completeness":
        return Completeness()
    elif criterion_string == "expected_fitness":
        raise Exception("Not implemented yet")
    else:
        raise Exception(f"The criterion string {criterion_string} was not recognised")


high_fitness = {"which": "high_fitness"}
low_fitness = {"which": "low_fitness"}
consistency = {"which": "consistent_fitness"}
explainability = {"which": "explainability"}
