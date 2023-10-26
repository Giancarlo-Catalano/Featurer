import datetime
import json
import time
from json import JSONDecodeError
from typing import Callable

from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem, TestableCombinatorialProblem
from SearchSpace import Candidate
from Version_E.InterestingAlgorithms.Miner import FeatureSelector, run_with_limited_budget, run_until_found_features, \
    FeatureMiner
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Testing import Miners, Criteria, Problems, Tests

JSON = dict
Settings = dict
TestResults = dict

TerminationPredicate = Callable

def to_json_object(input_string: str) -> JSON:
    return json.loads(input_string)


def get_random_candidates_and_fitnesses(problem: CombinatorialProblem,
                                        sample_size) -> (list[Candidate], list[float]):
    random_candidates = [problem.get_random_candidate_solution() for _ in range(sample_size)]

    scores = [problem.score_of_candidate(c) for c in random_candidates]
    return random_candidates, scores


def error_result(description) -> JSON:
    return {"error": description}


def execute_and_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


def run_multiple_times(func, runs, *args, **kwargs):
    return [func(*args, **kwargs) for _ in range(runs)]


def decode_problem(problem_parameters: Settings) -> CombinatorialProblem:
    return Problems.decode_problem(problem_parameters)


def decode_criterion(criterion_parameters: Settings, problem: CombinatorialProblem) -> MeasurableCriterion:
    return Criteria.decode_criterion(criterion_parameters, problem)


def decode_sample_size(arguments: Settings) -> int:
    return arguments["test"]["sample_size"]


def make_selector(problem: CombinatorialProblem, sample_size: int, criterion: MeasurableCriterion) -> FeatureSelector:
    training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)
    selector = FeatureSelector(training_ppi, criterion)
    return selector


def decode_miner(miner_arguments: Settings, selector, termination_predicate: TerminationPredicate) -> FeatureMiner:
    return Miners.decode_miner(miner_arguments, selector, termination_predicate)


def generate_problem_miner(arguments: Settings, overloading_miner_arguments=None) -> (
        CombinatorialProblem, FeatureMiner):
    problem = Problems.decode_problem(arguments["problem"])
    criterion = Criteria.decode_criterion(arguments["criterion"], problem)
    sample_size = arguments["test"]["sample_size"]
    training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)
    selector = FeatureSelector(training_ppi, criterion)

    if overloading_miner_arguments is None:
        overloading_miner_arguments = arguments["miner"]
    miner = Miners.decode_miner(overloading_miner_arguments, selector)
    return problem, miner


def run_test(arguments: dict):
    result_json = Tests.apply_test(arguments)  # important part
    output_json = {"parameters": arguments, "result": result_json}
    print(json.dumps(output_json))



