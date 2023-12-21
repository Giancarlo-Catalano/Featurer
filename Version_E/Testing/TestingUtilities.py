import json
import sys
import time
from typing import Callable

from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem, TestableCombinatorialProblem
from SearchSpace import Candidate
from Version_E.InterestingAlgorithms.Miner import FeatureSelector, run_with_limited_budget, run_until_found_features, \
    FeatureMiner
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Sampling.GASampler import GASampler
from Version_E.Testing import Miners, Criteria, Problems

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


def get_evolved_population_sample(problem: CombinatorialProblem,
                                  population_size: int,
                                  evaluation_budget: int) -> PrecomputedPopulationInformation:
    ga = GASampler(problem.score_of_candidate,
                   problem.search_space,
                   population_size,
                   run_with_limited_budget(evaluation_budget))
    population = ga.evolve_population()
    fitness_list = [problem.score_of_candidate(candidate) for candidate in population]

    return PrecomputedPopulationInformation(problem.search_space, population, fitness_list)


def decode_termination_predicate(problem: CombinatorialProblem, test_settings: Settings) -> TerminationPredicate:
    test_kind = test_settings["which"]

    if test_kind == "run_with_limited_budget":
        budget: int = test_settings["budget"]
        return run_with_limited_budget(budget)
    elif test_kind == "run_until_success":
        budget: int = test_settings["budget"]
        assert (
            isinstance(problem, TestableCombinatorialProblem))  # we assume that it is a TestableCombinatorialProblem
        target_individuals = problem.get_ideal_features()
        return run_until_found_features(target_individuals, max_budget=budget)
    else:
        raise Exception(f"Could not generate a termination function for the following test settings: {test_settings}")


def decode_problem(problem_arguments: Settings):
    return Problems.decode_problem(problem_arguments)


def decode_criterion(criterion_arguments: Settings, problem: CombinatorialProblem) -> MeasurableCriterion:
    return Criteria.decode_criterion(criterion_arguments, problem)


def make_selector(problem: CombinatorialProblem, sample_size: int, ga_budget: int,
                  criterion: MeasurableCriterion) -> FeatureSelector:
    training_ppi = get_evolved_population_sample(problem, sample_size, ga_budget)
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
    miner = Miners.decode_miner(overloading_miner_arguments, selector,
                                decode_termination_predicate(problem, arguments["test"]))
    return problem, miner


def load_json_from_file(file_name: str) -> list[dict]:
    try:
        with open(file_name, 'r') as file:
            return json.load(file)

    except IOError:
        print(f"Could not read file: {file_name}")


def load_data_from_second_file():
    second_argument = sys.argv[2]
    return load_json_from_file(second_argument)
