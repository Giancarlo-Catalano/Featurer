import datetime
import json
from json import JSONDecodeError

from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from SearchSpace import Candidate
from Version_E.InterestingAlgorithms.Miner import FeatureSelector
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Testing import Miners, Criteria, Problems, Tests

JSON = dict


def to_json_object(input_string: str) -> JSON:
    return json.loads(input_string)


def get_random_candidates_and_fitnesses(problem: CombinatorialProblem,
                                        sample_size) -> (list[Candidate], list[float]):
    random_candidates = [problem.get_random_candidate_solution() for _ in range(sample_size)]

    scores = [problem.score_of_candidate(c) for c in random_candidates]
    return random_candidates, scores

def error_result(description) -> JSON:
    return {"error": description}


def run_test(arguments: dict):
    result_json = Tests.apply_test(arguments)  # important part
    output_json = {"parameters": arguments, "result": result_json}
    print(json.dumps(output_json))



