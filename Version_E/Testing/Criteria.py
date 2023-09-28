from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from Version_E.MeasurableCriterion.CriterionUtilities import All, Any, Not, Balance
from Version_E.MeasurableCriterion.Explainability import Explainability
from Version_E.MeasurableCriterion.GoodFitness import HighFitness, ConsistentFitness
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.MeasurableCriterion.Popularity import Overrepresentation, Commonality
from Version_E.MeasurableCriterion.Robustness import Robustness


def decode_criterion(properties: dict, problem: CombinatorialProblem) -> MeasurableCriterion:
    criterion_string = properties["which"]

    if criterion_string == "explainability":
        return Explainability(problem)
    elif criterion_string == "high_fitness":
        return HighFitness()
    elif criterion_string == "low_fitness":
        return Not(HighFitness())
    elif criterion_string == "consistent_fitness":
        return ConsistentFitness()
    elif criterion_string == "overrepresentation":
        return Overrepresentation()
    elif criterion_string == "underrepresentation":
        return Not(Overrepresentation())
    elif criterion_string == "commonality":
        return Commonality()
    elif criterion_string == "robustness":
        return Robustness(properties["min_diff"], "max_diff")
    elif criterion_string == "not":
        return Not(decode_criterion(properties["argument"], problem))
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


high_fitness = {"which": "high_fitness"}
low_fitness = {"which": "low_fitness"}
consistency = {"which": "consistent_fitness"}
explainability = {"which": "explainability"}

consistently_high_fitness = {"which": "balance",
                             "arguments": [high_fitness, consistency],
                             "weights": [2, 1]}

consistently_low_fitness = {"which": "balance",
                            "arguments": [low_fitness, consistency],
                            "weights": [2, 1]}

low_fitness_and_explainable = {"which": "balance",
                               "arguments": [consistently_low_fitness, explainability],
                               "weights": [2, 1]}

high_fitness_and_explainable = {"which": "balance",
                               "arguments": [consistently_high_fitness, explainability],
                               "weights": [2, 1]}
